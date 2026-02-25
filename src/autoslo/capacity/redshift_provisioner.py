"""
redshift_provisioner.py
-----------------------
Live cluster provisioner for AWS Redshift Serverless.

Wraps the provisioning functions from
``experiments/10_workgroup_initialization_benchmarking/workgroup_creation_benchmarking.py``
into the :class:`~autoslo.capacity.cluster_provisioner.ClusterProvisioner`
interface, enabling the online runner to dynamically spin up and tear down
Redshift Serverless workgroups.

.. note::

   This provisioner makes real AWS API calls and requires valid credentials.
   It is **not** used in simulation — use
   :class:`~autoslo.capacity.cluster_provisioner.SimulatedProvisioner` instead.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.capacity.cluster_provisioner import ClusterProvisioner

logger = logging.getLogger(__name__)

# Default constants (match workgroup_creation_benchmarking.py)
_DEFAULT_AWS_REGION = "us-east-1"
_DEFAULT_ADMIN_USERNAME = "admin"
_DEFAULT_ADMIN_PASSWORD = "Password123"
_DEFAULT_DATASHARE_ACCOUNT_ID = "147854383891"
_DEFAULT_DATASHARE_NAMESPACE_ID = "1015d398-b04c-40d0-bb67-257e0956c96d"
_DEFAULT_SCHEMA_SCALES = [1, 10, 100, 1000, 3000, 10000]
_DEFAULT_DB_NAME = "dev"
_DEFAULT_PORT = 5439


class RedshiftServerlessProvisioner(ClusterProvisioner):
    """Provisioner that creates / destroys AWS Redshift Serverless
    workgroups.

    Each :meth:`spin_up` call:

    1. Creates a namespace (``create_namespace``).
    2. Creates a workgroup with the requested RPU (``create_workgroup``).
    3. Waits for the workgroup to become ``AVAILABLE``.
    4. Attaches the TPC-DS datashare (``attach_tpcds_database``).
    5. Creates external schemas for all configured scale factors.

    The resulting ``Cluster`` has full ``conn_info`` and is ready for
    query routing.

    Parameters
    ----------
    aws_region :
        AWS region for all API calls.
    aws_account_id :
        AWS account ID (used to form the Redshift Serverless hostname).
    admin_username :
        Admin username for newly created namespaces.
    admin_password :
        Admin password for newly created namespaces.
    datashare_account_id :
        Account ID that owns the TPC-DS datashare.
    datashare_namespace_id :
        Namespace ID for the TPC-DS datashare.
    schema_scales :
        TPC-DS scale factors to create external schemas for.
    db_name :
        Database name for connections.
    port :
        Port for connections.
    """

    def __init__(
        self,
        aws_account_id: str,
        aws_region: str = _DEFAULT_AWS_REGION,
        admin_username: str = _DEFAULT_ADMIN_USERNAME,
        admin_password: str = _DEFAULT_ADMIN_PASSWORD,
        datashare_account_id: str = _DEFAULT_DATASHARE_ACCOUNT_ID,
        datashare_namespace_id: str = _DEFAULT_DATASHARE_NAMESPACE_ID,
        schema_scales: Optional[list[int]] = None,
        db_name: str = _DEFAULT_DB_NAME,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._aws_region = aws_region
        self._aws_account_id = aws_account_id
        self._admin_username = admin_username
        self._admin_password = admin_password
        self._datashare_account_id = datashare_account_id
        self._datashare_namespace_id = datashare_namespace_id
        self._schema_scales = (
            schema_scales if schema_scales is not None else _DEFAULT_SCHEMA_SCALES
        )
        self._db_name = db_name
        self._port = port

    # ------------------------------------------------------------------
    # Internal AWS helpers (thin wrappers — logic copied from
    # workgroup_creation_benchmarking.py to avoid import-path issues)
    # ------------------------------------------------------------------

    def _get_client(self, service: str):  # noqa: ANN201
        """Lazy boto3 client (import deferred so tests without boto3 skip)."""
        import boto3

        return boto3.client(service, region_name=self._aws_region)

    def _create_namespace(self, namespace_name: str) -> bool:
        client = self._get_client("redshift-serverless")
        try:
            params = {
                "namespaceName": namespace_name,
                "adminUsername": self._admin_username,
            }
            if self._admin_password:
                params["adminUserPassword"] = self._admin_password
            client.create_namespace(**params)
            logger.info("Namespace %s creation initiated.", namespace_name)
            return True
        except client.exceptions.ConflictException:
            logger.info("Namespace %s already exists.", namespace_name)
            return True
        except Exception:
            logger.exception("Namespace creation failed for %s", namespace_name)
            return False

    def _create_workgroup(
        self, workgroup_name: str, base_rpu: int, namespace_name: str
    ) -> bool:
        client = self._get_client("redshift-serverless")
        try:
            client.create_workgroup(
                workgroupName=workgroup_name,
                namespaceName=namespace_name,
                baseCapacity=base_rpu,
                maxCapacity=base_rpu,
                publiclyAccessible=True,
            )
            logger.info(
                "Workgroup %s (%d RPU) creation initiated.",
                workgroup_name,
                base_rpu,
            )
            return True
        except client.exceptions.ConflictException:
            logger.info("Workgroup %s already exists.", workgroup_name)
            return True
        except Exception:
            logger.exception(
                "Workgroup creation failed for %s", workgroup_name
            )
            return False

    def _wait_for_namespace_available(
        self,
        namespace_name: str,
        max_wait_seconds: int = 600,
        poll_interval: int = 5,
    ) -> bool:
        client = self._get_client("redshift-serverless")
        start = time.time()
        while time.time() - start < max_wait_seconds:
            try:
                resp = client.get_namespace(namespaceName=namespace_name)
                status = resp["namespace"]["status"]
                if status == "AVAILABLE":
                    logger.info(
                        "Namespace %s available (%.1fs).",
                        namespace_name,
                        time.time() - start,
                    )
                    return True
                logger.debug("Namespace %s status: %s", namespace_name, status)
                time.sleep(poll_interval)
            except Exception:
                logger.exception(
                    "Error checking namespace %s status", namespace_name
                )
                return False
        logger.error(
            "Timeout waiting for namespace %s to become available.",
            namespace_name,
        )
        return False

    def _wait_for_workgroup_available(
        self,
        workgroup_name: str,
        max_wait_seconds: int = 600,
        poll_interval: int = 5,
    ) -> bool:
        client = self._get_client("redshift-serverless")
        start = time.time()
        while time.time() - start < max_wait_seconds:
            try:
                resp = client.get_workgroup(workgroupName=workgroup_name)
                status = resp["workgroup"]["status"]
                if status == "AVAILABLE":
                    logger.info(
                        "Workgroup %s available (%.1fs).",
                        workgroup_name,
                        time.time() - start,
                    )
                    return True
                logger.debug("Workgroup %s status: %s", workgroup_name, status)
                time.sleep(poll_interval)
            except Exception:
                logger.exception(
                    "Error checking workgroup %s status", workgroup_name
                )
                return False
        logger.error(
            "Timeout waiting for workgroup %s to become available.",
            workgroup_name,
        )
        return False

    def _attach_tpcds_database(self, workgroup_name: str) -> bool:
        client = self._get_client("redshift-data")
        sql = (
            f"CREATE DATABASE tpcds_db "
            f"FROM DATASHARE tpcds_datashare "
            f"OF ACCOUNT '{self._datashare_account_id}' "
            f"NAMESPACE '{self._datashare_namespace_id}';"
        )
        try:
            resp = client.execute_statement(
                WorkgroupName=workgroup_name, Sql=sql, Database=self._db_name
            )
            stmt_id = resp["Id"]
        except Exception:
            logger.exception("Datashare attach submit failed for %s", workgroup_name)
            return False

        return self._wait_for_statement(client, stmt_id, "attach_tpcds")

    def _create_external_schemas(self, workgroup_name: str) -> bool:
        client = self._get_client("redshift-data")
        all_ok = True
        for scale in self._schema_scales:
            sql = (
                f"CREATE EXTERNAL SCHEMA ext_tpcds{scale} "
                f"FROM redshift DATABASE tpcds_db SCHEMA tpcds{scale};"
            )
            try:
                resp = client.execute_statement(
                    WorkgroupName=workgroup_name,
                    Sql=sql,
                    Database=self._db_name,
                )
                stmt_id = resp["Id"]
            except Exception:
                logger.exception(
                    "Schema ext_tpcds%d submit failed for %s",
                    scale,
                    workgroup_name,
                )
                all_ok = False
                continue
            if not self._wait_for_statement(
                client, stmt_id, f"ext_tpcds{scale}"
            ):
                all_ok = False
        return all_ok

    @staticmethod
    def _wait_for_statement(
        client,  # noqa: ANN001
        statement_id: str,
        label: str,
        max_wait: int = 600,
        poll: int = 5,
    ) -> bool:
        start = time.time()
        while time.time() - start < max_wait:
            resp = client.describe_statement(Id=statement_id)
            status = resp["Status"]
            if status == "FINISHED":
                logger.info("Statement %s finished.", label)
                return True
            if status in ("FAILED", "ABORTED"):
                logger.error(
                    "Statement %s %s: %s",
                    label,
                    status,
                    resp.get("Error", "?"),
                )
                return False
            time.sleep(poll)
        logger.error("Statement %s timed out.", label)
        return False

    def _delete_workgroup(self, workgroup_name: str) -> bool:
        client = self._get_client("redshift-serverless")
        try:
            client.delete_workgroup(workgroupName=workgroup_name)
            logger.info("Workgroup %s deletion initiated.", workgroup_name)
            return True
        except Exception:
            logger.exception(
                "Workgroup deletion failed for %s", workgroup_name
            )
            return False

    def _wait_for_workgroup_deleted(
        self,
        workgroup_name: str,
        max_wait_seconds: int = 600,
        poll_interval: int = 5,
    ) -> bool:
        client = self._get_client("redshift-serverless")
        start = time.time()
        while time.time() - start < max_wait_seconds:
            try:
                client.get_workgroup(workgroupName=workgroup_name)
                time.sleep(poll_interval)
            except client.exceptions.ResourceNotFoundException:
                logger.info(
                    "Workgroup %s deleted (%.1fs).",
                    workgroup_name,
                    time.time() - start,
                )
                return True
            except Exception:
                logger.exception(
                    "Error checking workgroup %s deletion", workgroup_name
                )
                return False
        logger.error(
            "Timeout waiting for workgroup %s deletion.", workgroup_name
        )
        return False

    def _delete_namespace(self, namespace_name: str) -> bool:
        client = self._get_client("redshift-serverless")
        try:
            client.delete_namespace(namespaceName=namespace_name)
            logger.info("Namespace %s deletion initiated.", namespace_name)
            return True
        except Exception:
            logger.exception(
                "Namespace deletion failed for %s", namespace_name
            )
            return False

    def _wait_for_namespace_deleted(
        self,
        namespace_name: str,
        max_wait_seconds: int = 600,
        poll_interval: int = 5,
    ) -> bool:
        client = self._get_client("redshift-serverless")
        start = time.time()
        while time.time() - start < max_wait_seconds:
            try:
                client.get_namespace(namespaceName=namespace_name)
                time.sleep(poll_interval)
            except client.exceptions.ResourceNotFoundException:
                logger.info(
                    "Namespace %s deleted (%.1fs).",
                    namespace_name,
                    time.time() - start,
                )
                return True
            except Exception:
                logger.exception(
                    "Error checking namespace %s deletion", namespace_name
                )
                return False
        logger.error(
            "Timeout waiting for namespace %s deletion.", namespace_name
        )
        return False

    # ------------------------------------------------------------------
    # ClusterProvisioner interface
    # ------------------------------------------------------------------

    def _workgroup_name(self, rpu: int) -> str:
        """Generate a DNS-compatible, globally unique workgroup name."""
        return f"autoslo-wg-{rpu}rpu-{int(time.time())}"

    def spin_up(self, rpu: int, current_time_s: float) -> Cluster:
        """Create a Redshift Serverless workgroup and return a ready
        ``Cluster``.

        Steps:
          1. Create namespace (named same as workgroup).
          2. Create workgroup with requested RPU.
          3. Wait for ``AVAILABLE`` status.
          4. Attach TPC-DS datashare.
          5. Create external schemas.

        Returns
        -------
        A ``Cluster`` with populated ``conn_info``.

        Raises
        ------
        RuntimeError
            If any provisioning step fails.
        """
        wg_name = self._workgroup_name(rpu)
        ns_name = wg_name  # 1:1 namespace-to-workgroup

        logger.info(
            "Spinning up workgroup %s with %d RPU ...", wg_name, rpu
        )

        if not self._create_namespace(ns_name):
            raise RuntimeError(f"Failed to create namespace {ns_name}")

        if not self._wait_for_namespace_available(ns_name):
            raise RuntimeError(
                f"Namespace {ns_name} did not become available"
            )

        if not self._create_workgroup(wg_name, rpu, ns_name):
            raise RuntimeError(f"Failed to create workgroup {wg_name}")

        if not self._wait_for_workgroup_available(wg_name):
            raise RuntimeError(
                f"Workgroup {wg_name} did not become available"
            )

        if not self._attach_tpcds_database(wg_name):
            logger.warning(
                "TPC-DS datashare attach failed for %s — continuing.",
                wg_name,
            )

        if not self._create_external_schemas(wg_name):
            logger.warning(
                "Some external schemas failed for %s — continuing.",
                wg_name,
            )

        # Build Cluster with conn_info
        host = ClusterConnInfo.form_hostname(
            workgroup_name=wg_name,
            aws_account_id=self._aws_account_id,
            aws_region=self._aws_region,
        )
        conn_info = ClusterConnInfo(
            host=host,
            port=self._port,
            dbname=self._db_name,
            user=self._admin_username,
            password=self._admin_password,
        )

        cluster = Cluster(rpu=rpu, name=wg_name, conn_info=conn_info)
        logger.info("Workgroup %s (%d RPU) is ready.", wg_name, rpu)
        return cluster

    def tear_down(self, cluster_name: str, current_time_s: float) -> None:
        """Delete the workgroup and its namespace.

        Parameters
        ----------
        cluster_name :
            The workgroup (= namespace) name to delete.
        current_time_s :
            The current time for bookkeeping.

        Raises
        ------
        RuntimeError
            If deletion fails.
        """
        logger.info("Tearing down workgroup %s ...", cluster_name)

        if not self._delete_workgroup(cluster_name):
            raise RuntimeError(
                f"Failed to delete workgroup {cluster_name}"
            )
        if not self._wait_for_workgroup_deleted(cluster_name):
            raise RuntimeError(
                f"Workgroup {cluster_name} was not deleted in time"
            )

        if not self._delete_namespace(cluster_name):
            raise RuntimeError(
                f"Failed to delete namespace {cluster_name}"
            )
        if not self._wait_for_namespace_deleted(cluster_name):
            raise RuntimeError(
                f"Namespace {cluster_name} was not deleted in time"
            )

        logger.info("Workgroup %s fully torn down.", cluster_name)
