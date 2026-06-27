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

import itertools
import logging
import os
import time

import boto3  # type: ignore
import numpy as np
import yaml

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.clusters.cluster_provisioner import ClusterProvisioner
from autoslo.config.component_configs import ProvisionerConfig
from autoslo.filesystem.structured_events import (
    BaseStructuredEvent,
    EventType,
    wall_clock_utc,
)
from autoslo.filesystem.structured_log import emit_structured

logger = logging.getLogger(__name__)

# Default constants (match workgroup_creation_benchmarking.py)
_DEFAULT_AWS_REGION = "us-east-1"
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
    config :
        The provisioner configuration.
    """

    MAX_NAMESPACES: int = 25

    def __init__(self, config: ProvisionerConfig) -> None:
        self._config = config
        absolute_aws_config_path = os.path.join(
            pu.AUTOSLO_ROOT, config.aws_config_path
        )
        with open(absolute_aws_config_path) as f:
            cfg = yaml.safe_load(f)

        if "aws_account_id" not in cfg:
            raise ValueError(f"Missing required config key: aws_account_id")

        self._aws_account_id = cfg["aws_account_id"]
        self._aws_region = cfg.get("aws_region", _DEFAULT_AWS_REGION)
        self._admin_username = cfg.get("admin_username")
        self._admin_password = cfg.get("admin_password")
        self._datashare_account_id = cfg.get("datashare_account_id")
        self._datashare_namespace_id = cfg.get("datashare_namespace_id")
        self._schema_scales = cfg.get("schema_scales", _DEFAULT_SCHEMA_SCALES)
        self._db_name = cfg.get("db_name", _DEFAULT_DB_NAME)
        self._port = cfg.get("port", _DEFAULT_PORT)
        self._max_capacity_ratio = config.max_capacity_ratio
        self._price_performance_target_level = (
            config.price_performance_target_level
        )

        self._seq_counter = itertools.count()
        self._reference_time_s: float = 0.0
        self._cluster_cache_state_dim = config.cluster_cache_state_dim

    @property
    def reference_time_s(self) -> float:
        return self._reference_time_s

    @reference_time_s.setter
    def reference_time_s(self, value: float) -> None:
        self._reference_time_s = value

    # ------------------------------------------------------------------
    # Internal AWS helpers (thin wrappers — logic copied from
    # workgroup_creation_benchmarking.py to avoid import-path issues)
    # ------------------------------------------------------------------

    def _get_client(self, service: str):  # noqa: ANN201
        """Lazy boto3 client (import deferred so tests without boto3 skip)."""

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
        except client.exceptions.ServiceQuotaExceededException:
            logger.error(
                "Namespace quota exceeded (max %d) while creating %s. "
                "Delete unused namespaces or request a limit increase.",
                self.MAX_NAMESPACES,
                namespace_name,
            )
            return False
        except Exception:
            logger.exception("Namespace creation failed for %s", namespace_name)
            return False

    def _create_workgroup(
        self, workgroup_name: str, base_rpu: int, namespace_name: str
    ) -> bool:
        client = self._get_client("redshift-serverless")
        max_rpu = max(base_rpu, int(base_rpu * self._max_capacity_ratio))
        try:
            kwargs: dict = {
                "workgroupName": workgroup_name,
                "namespaceName": namespace_name,
                "publiclyAccessible": True,
                "maxCapacity": max_rpu,
            }
            if self._price_performance_target_level is None:
                kwargs["baseCapacity"] = base_rpu
            else:
                kwargs["pricePerformanceTarget"] = {
                    "level": self._price_performance_target_level,
                    "status": "ENABLED",
                }
            client.create_workgroup(**kwargs)
            log_message = f"Workgroup {workgroup_name} creation initiated with "
            if self._price_performance_target_level is None:
                log_message += f"base RPU {base_rpu} "
            else:
                log_message += (
                    f"price-performance target level "
                    f"{self._price_performance_target_level} "
                )
            log_message += f"and max RPU {max_rpu}."
            logger.info(log_message)
            return True
        except client.exceptions.ConflictException:
            logger.info("Workgroup %s already exists.", workgroup_name)
            return True
        except Exception:
            logger.exception("Workgroup creation failed for %s", workgroup_name)
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
            logger.exception(
                "Datashare attach submit failed for %s", workgroup_name
            )
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

    _DELETE_RETRY_DELAYS_S = (10, 30, 60)

    def _delete_workgroup(self, workgroup_name: str) -> tuple[bool, str | None]:
        """Delete a workgroup, returning ``(success, namespace_name)``.

        Retries on ``ConflictException`` (in-flight operation) with
        exponential backoff.
        """
        client = self._get_client("redshift-serverless")
        try:
            resp = client.get_workgroup(workgroupName=workgroup_name)
            namespace_name: str | None = resp["workgroup"].get("namespaceName")
        except Exception:
            logger.exception("Failed to look up workgroup %s", workgroup_name)
            return False, None

        last_exc: Exception | None = None
        for attempt, delay in enumerate(
            (0, *self._DELETE_RETRY_DELAYS_S), start=1
        ):
            if delay:
                logger.info(
                    "Retrying deletion of workgroup %s in %ds (attempt %d).",
                    workgroup_name,
                    delay,
                    attempt,
                )
                time.sleep(delay)
            try:
                client.delete_workgroup(workgroupName=workgroup_name)
                logger.info("Workgroup %s deletion initiated.", workgroup_name)
                return True, namespace_name
            except client.exceptions.ConflictException as exc:
                last_exc = exc
                logger.warning(
                    "ConflictException deleting workgroup %s (attempt %d): %s",
                    workgroup_name,
                    attempt,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Workgroup deletion failed for %s", workgroup_name
                )
                return False, namespace_name

        logger.error(
            "Workgroup %s deletion failed after %d attempts: %s",
            workgroup_name,
            len(self._DELETE_RETRY_DELAYS_S) + 1,
            last_exc,
        )
        return False, namespace_name

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
        """Delete a namespace.

        Retries on ``ConflictException`` with exponential backoff.
        """
        client = self._get_client("redshift-serverless")

        last_exc: Exception | None = None
        for attempt, delay in enumerate(
            (0, *self._DELETE_RETRY_DELAYS_S), start=1
        ):
            if delay:
                logger.info(
                    "Retrying deletion of namespace %s in %ds (attempt %d).",
                    namespace_name,
                    delay,
                    attempt,
                )
                time.sleep(delay)
            try:
                client.delete_namespace(namespaceName=namespace_name)
                logger.info("Namespace %s deletion initiated.", namespace_name)
                return True
            except client.exceptions.ResourceNotFoundException:
                logger.info(
                    "Namespace %s already gone — nothing to delete.",
                    namespace_name,
                )
                return True
            except client.exceptions.ConflictException as exc:
                last_exc = exc
                logger.warning(
                    "ConflictException deleting namespace %s (attempt %d): %s",
                    namespace_name,
                    attempt,
                    exc,
                )
            except Exception:
                logger.exception(
                    "Namespace deletion failed for %s", namespace_name
                )
                return False

        logger.error(
            "Namespace %s deletion failed after %d attempts: %s",
            namespace_name,
            len(self._DELETE_RETRY_DELAYS_S) + 1,
            last_exc,
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

    def _workgroup_and_namespace_names(self, rpu: int) -> tuple[str, str]:
        """Generate a DNS-compatible, globally unique workgroup name."""
        seq = next(self._seq_counter)

        wg_name = f"autoslo-{rpu}-{self._config.run_id}-{seq}"
        ns_name = f"autoslo-{rpu}-{self._config.run_id}-{seq}-ns"
        return wg_name, ns_name

    def _best_effort_cleanup(
        self, workgroup_name: str, namespace_name: str
    ) -> None:
        """Attempt to clean up a partially-created workgroup and namespace.

        Called when spin-up fails partway through.  Errors are logged
        but never raised — the caller will raise the original error.
        """
        logger.info(
            "Attempting best-effort cleanup of %s / %s after failed spin-up.",
            workgroup_name,
            namespace_name,
        )
        try:
            ok, _ = self._delete_workgroup(workgroup_name)
            if ok:
                self._wait_for_workgroup_deleted(workgroup_name)
        except Exception:
            logger.debug(
                "Cleanup: workgroup %s delete skipped (may not exist).",
                workgroup_name,
            )
        try:
            if self._delete_namespace(namespace_name):
                self._wait_for_namespace_deleted(namespace_name)
        except Exception:
            logger.debug(
                "Cleanup: namespace %s delete skipped (may not exist).",
                namespace_name,
            )

    def spin_up(self, rpu: int, rel_time_s: float) -> Cluster:
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
        spin_up_start = wall_clock_utc()
        wg_name, ns_name = self._workgroup_and_namespace_names(rpu)

        logger.info("Spinning up workgroup %s with %d RPU ...", wg_name, rpu)
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.SPIN_UP_STARTED,
                source="RedshiftServerlessProvisioner",
                cluster_name=wg_name,
            )
        )

        if not self._create_namespace(ns_name):
            raise RuntimeError(f"Failed to create namespace {ns_name}")

        if not self._wait_for_namespace_available(ns_name):
            self._best_effort_cleanup(wg_name, ns_name)
            raise RuntimeError(f"Namespace {ns_name} did not become available")

        if not self._create_workgroup(wg_name, rpu, ns_name):
            self._best_effort_cleanup(wg_name, ns_name)
            raise RuntimeError(f"Failed to create workgroup {wg_name}")

        if not self._wait_for_workgroup_available(wg_name):
            self._best_effort_cleanup(wg_name, ns_name)
            raise RuntimeError(f"Workgroup {wg_name} did not become available")

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
        now = wall_clock_utc()
        cluster = Cluster(
            creation_time_s=now - self._reference_time_s,
            rpu=rpu,
            cache_state=np.zeros(
                self._cluster_cache_state_dim, dtype=np.float32
            ),
            name=wg_name,
            conn_info=conn_info,
        )

        return cluster

    def tear_down(self, cluster_name: str, rel_time_s: float) -> None:
        """Delete the workgroup and its namespace.

        Parameters
        ----------
        cluster_name :
            The workgroup (= namespace) name to delete.
        rel_time_s :
            Relative time in seconds since run start.

        Raises
        ------
        RuntimeError
            If deletion fails.
        """
        tear_down_start = wall_clock_utc()
        logger.info("Tearing down workgroup %s ...", cluster_name)
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.TEAR_DOWN_STARTED,
                source="RedshiftServerlessProvisioner",
                cluster_name=cluster_name,
            )
        )

        ok, namespace_name = self._delete_workgroup(cluster_name)
        if not ok:
            raise RuntimeError(f"Failed to delete workgroup {cluster_name}")
        if not self._wait_for_workgroup_deleted(cluster_name):
            raise RuntimeError(
                f"Workgroup {cluster_name} was not deleted in time"
            )

        ns = namespace_name or (cluster_name + "-ns")
        if not self._delete_namespace(ns):
            raise RuntimeError(f"Failed to delete namespace {ns}")
        if not self._wait_for_namespace_deleted(ns):
            raise RuntimeError(f"Namespace {ns} was not deleted in time")
