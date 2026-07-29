import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import yaml

import autoslo.filesystem.path_utils as pu

# ============================================================================
# Configuration Constants
# ============================================================================
AWS_REGION = "us-east-1"
SCHEMA_SCALES = [1, 10, 100, 1000, 3000, 10000]
DATASHARE_ACCOUNT_ID = "147854383891"
DATASHARE_NAMESPACE_ID = "1015d398-b04c-40d0-bb67-257e0956c96d"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "Password123"
OUTPUT_DIR = (
    pu.AUTOSLO_ROOT
    / "experiments"
    / "10_workgroup_initialization_benchmarking"
    / "logs"
)
DEFAULT_POLL_INTERVAL = 1  # seconds


# ============================================================================
# Logging Setup
# ============================================================================
class DualWriter:
    """Write to both console and log file."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, message):
        self.console.write(message)
        self.console.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()


def setup_logging(log_path: Path) -> DualWriter:
    """Setup dual logging to console and file."""
    log_file = open(log_path, "w")
    dual_writer = DualWriter(sys.stdout, log_file)
    sys.stdout = dual_writer
    return dual_writer


def run_command(cmd: list[str], description: str) -> bool:
    """
    Execute a shell command and report status.

    Parameters:
        cmd: Command as list of strings for subprocess.
        description: Human-readable description of what's being done.

    Returns:
        True if successful, False otherwise.
    """
    print(f"{datetime.now()} {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{datetime.now()} ✓ {description} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{datetime.now()} ✗ {description} failed:")
        print(e.stderr)
        return False


# Initialize AWS clients
redshift_serverless_client = None
redshift_data_client = None


def get_clients():
    """Initialize AWS clients."""
    global redshift_serverless_client, redshift_data_client
    if redshift_serverless_client is None:
        redshift_serverless_client = boto3.client(
            "redshift-serverless", region_name=AWS_REGION
        )
    if redshift_data_client is None:
        redshift_data_client = boto3.client(
            "redshift-data", region_name=AWS_REGION
        )
    return redshift_serverless_client, redshift_data_client


def create_namespace(
    namespace_name: str,
    admin_username: str = "admin",
    admin_password: str = None,
    aws_region: str = "us-east-1",
) -> bool:
    """
    Create a Redshift Serverless namespace.

    Parameters:
        namespace_name: Name for the new namespace.
        admin_username: Admin username for the namespace.
        admin_password: Admin password for the namespace.
        aws_region: AWS region.

    Returns:
        True if successful, False otherwise.
    """
    print(f"{datetime.now()} Creating namespace '{namespace_name}'...")
    try:
        client = boto3.client("redshift-serverless", region_name=aws_region)
        params = {
            "namespaceName": namespace_name,
            "adminUsername": admin_username,
        }
        if admin_password:
            params["adminUserPassword"] = admin_password
        client.create_namespace(**params)
        print(f"{datetime.now()} ✓ Namespace creation initiated.")
        return True
    except client.exceptions.ConflictException:
        print(
            f"{datetime.now()} ⚠ Namespace '{namespace_name}' already exists, skipping creation."
        )
        return True
    except Exception as e:
        print(f"{datetime.now()} ✗ Namespace creation failed:")
        print(str(e))
        return False


def create_workgroup(
    workgroup_name: str,
    base_rpu: int,
    namespace_name: str,
    aws_region: str = "us-east-1",
) -> bool:
    """
    Create a Redshift Serverless workgroup with specified RPU.

    Parameters:
        workgroup_name: Name for the new workgroup.
        base_rpu: Base RPU for the workgroup (also sets max RPU).
        namespace_name: Name of the namespace to attach to.
        aws_region: AWS region for the workgroup.

    Returns:
        True if successful, False otherwise.
    """
    print(
        f"{datetime.now()} Creating workgroup '{workgroup_name}' with {base_rpu} RPU..."
    )
    try:
        client = boto3.client("redshift-serverless", region_name=aws_region)
        client.create_workgroup(
            workgroupName=workgroup_name,
            namespaceName=namespace_name,
            baseCapacity=base_rpu,
            maxCapacity=base_rpu,
            publiclyAccessible=True,
        )
        print(
            f"{datetime.now()} ✓ Workgroup creation initiated (publicly accessible)."
        )
        return True
    except client.exceptions.ConflictException:
        print(
            f"{datetime.now()} ⚠ Workgroup '{workgroup_name}' already exists, skipping creation."
        )
        return True
    except Exception as e:
        print(f"{datetime.now()} ✗ Workgroup creation failed:")
        print(str(e))
        return False


def make_workgroup_publicly_accessible(
    workgroup_name: str,
    aws_region: str = "us-east-1",
) -> bool:
    """
    Make a Redshift Serverless workgroup publicly accessible.

    Parameters:
        workgroup_name: Name of the workgroup.
        aws_region: AWS region.

    Returns:
        True if successful, False otherwise.
    """
    print(
        f"{datetime.now()} Making workgroup '{workgroup_name}' publicly accessible..."
    )
    try:
        client = boto3.client("redshift-serverless", region_name=aws_region)
        client.update_workgroup(
            workgroupName=workgroup_name,
            publiclyAccessible=True,
        )
        print(f"{datetime.now()} ✓ Workgroup made publicly accessible.")
        return True
    except client.exceptions.ValidationException as e:
        if "already public" in str(e):
            print(
                f"{datetime.now()} ⚠ Workgroup '{workgroup_name}' is already publicly accessible."
            )
            return True
        print(
            f"{datetime.now()} ✗ Failed to make workgroup publicly accessible:"
        )
        print(str(e))
        return False
    except Exception as e:
        print(
            f"{datetime.now()} ✗ Failed to make workgroup publicly accessible:"
        )
        print(str(e))
        return False


def wait_for_workgroup_available(
    workgroup_name: str,
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """
    Wait for a Redshift Serverless workgroup to become available.

    Parameters:
        workgroup_name: Name of the workgroup.
        aws_region: AWS region.
        max_wait_seconds: Maximum time to wait in seconds.
        poll_interval: Time between status checks in seconds.

    Returns:
        True if workgroup becomes available, False if timeout or error.
    """
    print(
        f"{datetime.now()} Waiting for workgroup '{workgroup_name}' to become available..."
    )
    start_time = datetime.now()

    client = boto3.client("redshift-serverless", region_name=aws_region)
    while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
        try:
            response = client.get_workgroup(workgroupName=workgroup_name)
            status = response["workgroup"]["status"]

            if status == "AVAILABLE":
                elapsed = datetime.now() - start_time
                print(
                    f"{datetime.now()} ✓ Workgroup is available (waited {elapsed})"
                )
                return True

            print(f"{datetime.now()} Workgroup status: {status}, waiting...")
            time.sleep(poll_interval)

        except Exception as e:
            print(f"{datetime.now()} ✗ Error checking workgroup status:")
            print(str(e))
            return False

    print(
        f"{datetime.now()} ✗ Timeout waiting for workgroup to become available"
    )
    return False


def wait_for_namespace_available(
    namespace_name: str,
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """
    Wait for a Redshift Serverless namespace to become available.

    Parameters:
        namespace_name: Name of the namespace.
        aws_region: AWS region.
        max_wait_seconds: Maximum time to wait in seconds.
        poll_interval: Time between status checks in seconds.

    Returns:
        True if namespace becomes available, False if timeout or error.
    """
    print(
        f"{datetime.now()} Waiting for namespace '{namespace_name}' to become available..."
    )
    start_time = datetime.now()

    client = boto3.client("redshift-serverless", region_name=aws_region)
    while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
        try:
            response = client.get_namespace(namespaceName=namespace_name)
            status = response["namespace"]["status"]

            if status == "AVAILABLE":
                elapsed = datetime.now() - start_time
                print(
                    f"{datetime.now()} ✓ Namespace is available (waited {elapsed})"
                )
                return True

            print(f"{datetime.now()} Namespace status: {status}, waiting...")
            time.sleep(poll_interval)

        except Exception as e:
            print(f"{datetime.now()} ✗ Error checking namespace status:")
            print(str(e))
            return False

    print(
        f"{datetime.now()} ✗ Timeout waiting for namespace to become available"
    )
    return False


def attach_tpcds_database(
    workgroup_name: str,
    datashare_account_id: str = "147854383891",
    namespace_id: str = "1015d398-b04c-40d0-bb67-257e0956c96d",
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """
    Attach TPC-DS database via datashare, with polling for statement completion.

    Parameters:
        workgroup_name: Name of the workgroup.
        datashare_account_id: AWS account ID owning the TPC-DS datashare.
        namespace_id: Redshift namespace ID.
        aws_region: AWS region.
        max_wait_seconds: Maximum time to wait for statement completion.
        poll_interval: Time between status checks in seconds.

    Returns:
        True if successful, False otherwise.
    """
    sql_command = f"""
    CREATE DATABASE tpcds_db
    FROM DATASHARE tpcds_datashare
    OF ACCOUNT '{datashare_account_id}'
    NAMESPACE '{namespace_id}';
    """

    print(f"{datetime.now()} Attempting to attach TPC-DS database...")
    start_time = datetime.now()

    client = boto3.client("redshift-data", region_name=aws_region)

    try:
        response = client.execute_statement(
            WorkgroupName=workgroup_name,
            Sql=sql_command,
            Database="dev",
        )
        statement_id = response["Id"]
        print(f"{datetime.now()} Statement submitted, ID: {statement_id}")
    except Exception as e:
        print(f"{datetime.now()} ✗ Error submitting TPC-DS database creation:")
        print(str(e))
        return False

    # Poll for statement completion
    while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
        try:
            response = client.describe_statement(Id=statement_id)
            status = response["Status"]

            if status == "FINISHED":
                print(
                    f"{datetime.now()} ✓ Successfully attached TPC-DS database."
                )
                return True
            elif status == "FAILED":
                error = response.get("Error", "Unknown error")
                print(f"{datetime.now()} ✗ Statement failed: {error}")
                return False
            elif status == "ABORTED":
                print(f"{datetime.now()} ✗ Statement was aborted.")
                return False

            print(f"{datetime.now()} Statement status: {status}, waiting...")
            time.sleep(poll_interval)

        except Exception as e:
            print(f"{datetime.now()} ✗ Error checking statement status:")
            print(str(e))
            return False

    print(f"{datetime.now()} ✗ Timeout waiting for TPC-DS database attachment")
    return False


def create_external_schemas(
    workgroup_name: str,
    schema_scales: list[int] = [1, 10, 100, 1000, 3000, 10000],
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """
    Create external schemas for specified TPC-DS scale factors.

    Parameters:
        workgroup_name: Name of the workgroup.
        schema_scales: List of TPC-DS scale factors to create schemas for.
        aws_region: AWS region.
        max_wait_seconds: Maximum time to wait for each statement.
        poll_interval: Time between status checks in seconds.

    Returns:
        True if successful, False otherwise.
    """
    all_success = True
    client = boto3.client("redshift-data", region_name=aws_region)

    for scale in schema_scales:
        sql_command = (
            f"CREATE EXTERNAL SCHEMA ext_tpcds{scale} "
            f"FROM redshift DATABASE tpcds_db SCHEMA tpcds{scale};"
        )

        print(f"{datetime.now()} Creating external schema ext_tpcds{scale}...")
        try:
            response = client.execute_statement(
                WorkgroupName=workgroup_name,
                Sql=sql_command,
                Database="dev",
            )
            statement_id = response["Id"]
        except Exception as e:
            print(f"{datetime.now()} ✗ Error submitting schema creation:")
            print(str(e))
            all_success = False
            continue

        # Poll for completion
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
            try:
                response = client.describe_statement(Id=statement_id)
                status = response["Status"]

                if status == "FINISHED":
                    print(
                        f"{datetime.now()} ✓ External schema ext_tpcds{scale} created."
                    )
                    break
                elif status in ("FAILED", "ABORTED"):
                    error = response.get("Error", "Unknown error")
                    print(f"{datetime.now()} ✗ Schema creation failed: {error}")
                    all_success = False
                    break

                time.sleep(poll_interval)

            except Exception as e:
                print(f"{datetime.now()} ✗ Error checking statement status:")
                print(str(e))
                all_success = False
                break
        else:
            print(
                f"{datetime.now()} ✗ Timeout creating external schema ext_tpcds{scale}"
            )
            all_success = False

    return all_success


def delete_workgroup(
    workgroup_name: str,
    aws_region: str = "us-east-1",
) -> bool:
    """Delete a Redshift Serverless workgroup."""
    print(f"{datetime.now()} Deleting workgroup '{workgroup_name}'...")
    try:
        client = boto3.client("redshift-serverless", region_name=aws_region)
        client.delete_workgroup(workgroupName=workgroup_name)
        print(f"{datetime.now()} ✓ Workgroup deletion initiated.")
        return True
    except Exception as e:
        print(f"{datetime.now()} ✗ Workgroup deletion failed:")
        print(str(e))
        return False


def wait_for_workgroup_deleted(
    workgroup_name: str,
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """Wait for a workgroup to be deleted."""
    print(
        f"{datetime.now()} Waiting for workgroup '{workgroup_name}' to be deleted..."
    )
    start_time = datetime.now()
    client = boto3.client("redshift-serverless", region_name=aws_region)

    while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
        try:
            client.get_workgroup(workgroupName=workgroup_name)
            print(f"{datetime.now()} Workgroup still exists, waiting...")
            time.sleep(poll_interval)
        except client.exceptions.ResourceNotFoundException:
            elapsed = datetime.now() - start_time
            print(f"{datetime.now()} ✓ Workgroup deleted (waited {elapsed})")
            return True
        except Exception as e:
            print(
                f"{datetime.now()} ✗ Error checking workgroup deletion status:"
            )
            print(str(e))
            return False

    print(f"{datetime.now()} ✗ Timeout waiting for workgroup deletion")
    return False


def delete_namespace(
    namespace_name: str,
    aws_region: str = "us-east-1",
) -> bool:
    """Delete a Redshift Serverless namespace."""
    print(f"{datetime.now()} Deleting namespace '{namespace_name}'...")
    try:
        client = boto3.client("redshift-serverless", region_name=aws_region)
        client.delete_namespace(namespaceName=namespace_name)
        print(f"{datetime.now()} ✓ Namespace deletion initiated.")
        return True
    except Exception as e:
        print(f"{datetime.now()} ✗ Namespace deletion failed:")
        print(str(e))
        return False


def wait_for_namespace_deleted(
    namespace_name: str,
    aws_region: str = "us-east-1",
    max_wait_seconds: int = 600,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """Wait for a namespace to be deleted."""
    print(
        f"{datetime.now()} Waiting for namespace '{namespace_name}' to be deleted..."
    )
    start_time = datetime.now()
    client = boto3.client("redshift-serverless", region_name=aws_region)

    while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
        try:
            client.get_namespace(namespaceName=namespace_name)
            print(f"{datetime.now()} Namespace still exists, waiting...")
            time.sleep(poll_interval)
        except client.exceptions.ResourceNotFoundException:
            elapsed = datetime.now() - start_time
            print(f"{datetime.now()} ✓ Namespace deleted (waited {elapsed})")
            return True
        except Exception as e:
            print(
                f"{datetime.now()} ✗ Error checking namespace deletion status:"
            )
            print(str(e))
            return False

    print(f"{datetime.now()} ✗ Timeout waiting for namespace deletion")
    return False


def _record_step(
    step_results: list[dict],
    step_name: str,
    start_time: datetime,
    end_time: datetime,
    success: bool,
    skipped: bool = False,
) -> None:
    step_results.append(
        {
            "step": step_name,
            "start": None if skipped else start_time.isoformat(),
            "end": None if skipped else end_time.isoformat(),
            "duration_seconds": (
                None
                if skipped
                else round((end_time - start_time).total_seconds(), 2)
            ),
            "success": success,
            "skipped": skipped,
        }
    )


def run_step(
    step_results: list[dict], step_name: str, func, *args, **kwargs
) -> bool:
    step_start = datetime.now()
    success = func(*args, **kwargs)
    _record_step(step_results, step_name, step_start, datetime.now(), success)
    return success


def mark_step_skipped(step_results: list[dict], step_name: str) -> None:
    now = datetime.now()
    _record_step(step_results, step_name, now, now, success=True, skipped=True)


def write_step_outputs(
    step_results: list[dict],
    log_path: Path,
    summary_path: Path,
    total_elapsed,
    meta: dict,
) -> None:
    log_lines = []
    for step in step_results:
        status = (
            "SKIPPED"
            if step.get("skipped")
            else ("OK" if step.get("success") else "FAIL")
        )
        log_lines.append(
            f"{step['step']}: {status} | start={step.get('start')} | end={step.get('end')} | duration={step.get('duration_seconds')}s"
        )
    log_lines.append(f"Total elapsed: {total_elapsed}")
    log_path.write_text("\n".join(log_lines))
    summary = {
        "meta": meta,
        "total_elapsed_seconds": round(total_elapsed.total_seconds(), 2),
        "steps": step_results,
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"{datetime.now()} Wrote log to {log_path}")
    print(f"{datetime.now()} Wrote summary to {summary_path}")


def setup_workgroup(
    workgroup_name: str,
    base_rpu: int,
    aws_region: str = "us-east-1",
    schema_scales: Optional[list[int]] = None,
    skip_workgroup_creation: bool = False,
    skip_datashare_attachment: bool = False,
    skip_workgroup_deletion: bool = False,
    namespace_name: Optional[str] = None,
    admin_username: str = "admin",
    admin_password: Optional[str] = None,
    output_dir: Path = Path("./"),
) -> bool:
    """
    Complete setup of a Redshift Serverless workgroup with TPC-DS schemas.

    Parameters:
        workgroup_name: Name for the workgroup.
        base_rpu: Base RPU for the workgroup.
        aws_region: AWS region.
        schema_scales: List of TPC-DS scale factors to create (default: [1, 10, 100, 1000, 3000, 10000]).
        skip_workgroup_creation: If True, skip creating the workgroup (assume it exists).
        skip_datashare_attachment: If True, skip attaching the TPC-DS datashare and creating external schemas.
        skip_workgroup_deletion: If True, skip deleting the workgroup and namespace after setup.
        namespace_name: Name for the namespace (defaults to workgroup_name + '-ns').
        admin_username: Admin username for the namespace.
        admin_password: Admin password for the namespace.
        output_dir: Directory to write log and summary files.

    Returns:
        True if all steps successful, False otherwise.
    """
    if schema_scales is None:
        schema_scales = [1, 10, 100, 1000, 3000, 10000]

    if namespace_name is None:
        namespace_name = f"{workgroup_name}-ns"

    print(f"\n{'='*70}")
    print(f"Setting up Redshift Serverless workgroup: {workgroup_name}")
    print(f"Namespace: {namespace_name}")
    print(f"Base RPU: {base_rpu}")
    print(f"Region: {aws_region}")
    print(f"Schema scales: {schema_scales}")
    print(f"{'='*70}\n")

    start_time = datetime.now()
    base_path = output_dir.expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    log_path = base_path / f"{workgroup_name}_setup_{timestamp}.log"
    summary_path = base_path / f"{workgroup_name}_setup_{timestamp}.yml"
    step_results: list[dict] = []

    def flush_and_return(success: bool) -> bool:
        total_elapsed = datetime.now() - start_time
        meta = {
            "workgroup_name": workgroup_name,
            "namespace_name": namespace_name,
            "aws_region": aws_region,
            "base_rpu": base_rpu,
            "schema_scales": schema_scales,
            "skip_workgroup_creation": skip_workgroup_creation,
            "output_dir": str(base_path),
        }
        write_step_outputs(
            step_results, log_path, summary_path, total_elapsed, meta
        )
        return success

    if not skip_workgroup_creation:
        if not run_step(
            step_results,
            "create_namespace",
            create_namespace,
            namespace_name,
            admin_username,
            admin_password,
            aws_region,
        ):
            print(f"\n✗ Namespace creation failed.")
            return flush_and_return(False)

        if not run_step(
            step_results,
            "wait_for_namespace_available",
            wait_for_namespace_available,
            namespace_name,
            aws_region,
        ):
            print(f"\n✗ Namespace did not become available in time.")
            return flush_and_return(False)

        if not run_step(
            step_results,
            "create_workgroup",
            create_workgroup,
            workgroup_name,
            base_rpu,
            namespace_name,
            aws_region,
        ):
            print(f"\n✗ Workgroup creation failed.")
            return flush_and_return(False)

        if not run_step(
            step_results,
            "wait_for_workgroup_available",
            wait_for_workgroup_available,
            workgroup_name,
            aws_region,
        ):
            print(f"\n✗ Workgroup did not become available in time.")
            return flush_and_return(False)
    else:
        for skipped_step in [
            "create_namespace",
            "wait_for_namespace_available",
            "create_workgroup",
            "wait_for_workgroup_available",
        ]:
            mark_step_skipped(step_results, skipped_step)

    if not skip_datashare_attachment:
        if not run_step(
            step_results,
            "attach_tpcds_database",
            attach_tpcds_database,
            workgroup_name,
            aws_region=aws_region,
        ):
            print(f"\n✗ Attaching TPC-DS database failed.")
            return flush_and_return(False)

        if not run_step(
            step_results,
            "create_external_schemas",
            create_external_schemas,
            workgroup_name,
            schema_scales,
            aws_region,
        ):
            print(f"\n✗ Creating external schemas failed.")
            return flush_and_return(False)
    else:
        for skipped_step in [
            "attach_tpcds_database",
            "create_external_schemas",
        ]:
            mark_step_skipped(step_results, skipped_step)

    # Pause before cleanup
    if not skip_workgroup_deletion:
        run_step(step_results, "wait_before_deletion", time.sleep, 180)
        if not run_step(
            step_results,
            "delete_workgroup",
            delete_workgroup,
            workgroup_name,
            aws_region,
        ):
            print(f"\n✗ Workgroup deletion failed.")
            return flush_and_return(False)
        if not run_step(
            step_results,
            "wait_for_workgroup_deleted",
            wait_for_workgroup_deleted,
            workgroup_name,
            aws_region,
        ):
            print(f"\n✗ Workgroup did not delete in time.")
            return flush_and_return(False)
        if not run_step(
            step_results,
            "delete_namespace",
            delete_namespace,
            namespace_name,
            aws_region,
        ):
            print(f"\n✗ Namespace deletion failed.")
            return flush_and_return(False)
        if not run_step(
            step_results,
            "wait_for_namespace_deleted",
            wait_for_namespace_deleted,
            namespace_name,
            aws_region,
        ):
            print(f"\n✗ Namespace did not delete in time.")
            return flush_and_return(False)
    else:
        for skipped_step in [
            "wait_before_deletion",
            "delete_workgroup",
            "wait_for_workgroup_deleted",
            "delete_namespace",
            "wait_for_namespace_deleted",
        ]:
            mark_step_skipped(step_results, skipped_step)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Workgroup setup completed successfully!")
    print(f"Total time: {elapsed}")
    print(f"{'='*70}\n")

    return flush_and_return(True)


def benchmark_workgroup_creation(
    base_workgroup_name: str,
    iterations_per_rpu: int = 5,
    min_rpu: int = 4,
    max_rpu: int = 256,
) -> None:
    """
    Benchmark workgroup creation/deletion across power-of-2 RPU sizes.

    Parameters:
        base_workgroup_name: Base name for workgroups (RPU and iteration appended).
        iterations_per_rpu: Number of create/delete cycles per RPU size.
        min_rpu: Minimum RPU (power of 2).
        max_rpu: Maximum RPU (power of 2).
    """
    base_path = Path(OUTPUT_DIR).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    benchmark_start = datetime.now()
    benchmark_timestamp = benchmark_start.strftime("%Y%m%d_%H%M%S")
    benchmark_log_path = base_path / f"benchmark_{benchmark_timestamp}.log"
    benchmark_results_path = (
        base_path / f"benchmark_results_{benchmark_timestamp}.yml"
    )

    # Setup logging
    dual_writer = setup_logging(benchmark_log_path)

    all_results = {
        "benchmark_start": benchmark_start.isoformat(),
        "base_workgroup_name": base_workgroup_name,
        "aws_region": AWS_REGION,
        "iterations_per_rpu": iterations_per_rpu,
        "rpu_results": {},
    }

    # Generate power-of-2 RPU sizes
    rpu_sizes = []
    rpu = min_rpu
    while rpu <= max_rpu:
        rpu_sizes.append(rpu)
        rpu *= 2

    print(f"\n{'='*70}")
    print(f"Benchmarking Redshift Serverless workgroup creation/deletion")
    print(f"RPU sizes: {rpu_sizes}")
    print(f"Iterations per RPU: {iterations_per_rpu}")
    print(f"{'='*70}\n")

    # Create list of all trials and randomize order
    trials = []
    for rpu in rpu_sizes:
        for iteration in range(1, iterations_per_rpu + 1):
            trials.append((rpu, iteration))

    random.seed(42)
    random.shuffle(trials)

    print(f"Trial execution order (randomized):")
    for idx, (rpu, iteration) in enumerate(trials, 1):
        print(f"  {idx}. RPU {rpu}, Iteration {iteration}")
    print()

    rpu_iteration_results_by_rpu: dict[str, list] = {
        f"rpu_{rpu}": [] for rpu in rpu_sizes
    }

    # Execute trials in randomized order
    for trial_num, (rpu, iteration) in enumerate(trials, 1):
        workgroup_name = f"{base_workgroup_name}-rpu{rpu}-iter{iteration}"
        namespace_name = f"{workgroup_name}-ns"

        if trial_num > 1:
            # Wait 3 minutes before starting next trial (except on first)
            print(f"\nWaiting 3 minutes before trial {trial_num}...")
            time.sleep(180)

        print(
            f"\nTrial {trial_num}/{len(trials)}: RPU {rpu}, Iteration {iteration}"
        )
        success = setup_workgroup(
            workgroup_name=workgroup_name,
            base_rpu=rpu,
            aws_region=AWS_REGION,
            schema_scales=SCHEMA_SCALES,
            namespace_name=namespace_name,
            admin_username=ADMIN_USERNAME,
            admin_password=ADMIN_PASSWORD,
            output_dir=OUTPUT_DIR,
        )

        # Read the most recent summary file for this iteration
        summary_files = sorted(base_path.glob(f"{workgroup_name}_setup_*.yml"))
        if summary_files:
            latest_summary = summary_files[-1]
            with open(latest_summary) as f:
                iteration_data = yaml.safe_load(f)
            rpu_iteration_results_by_rpu[f"rpu_{rpu}"].append(
                {
                    "iteration": iteration,
                    "success": success,
                    "data": iteration_data,
                }
            )
            print(f"✓ Results saved to {latest_summary}")

        # Dump results after each iteration
        all_results["rpu_results"] = rpu_iteration_results_by_rpu
        with open(benchmark_results_path, "w") as f:
            yaml.safe_dump(all_results, f, sort_keys=False)
        print(f"✓ Benchmark results updated at {benchmark_results_path}")

    benchmark_end = datetime.now()
    all_results["benchmark_end"] = benchmark_end.isoformat()
    all_results["total_elapsed_seconds"] = round(
        (benchmark_end - benchmark_start).total_seconds(), 2
    )

    with open(benchmark_results_path, "w") as f:
        yaml.safe_dump(all_results, f, sort_keys=False)

    print(f"\n{'='*70}")
    print(f"✓ Benchmark completed!")
    print(f"Results: {benchmark_results_path}")
    print(f"Log: {benchmark_log_path}")
    print(f"Total time: {benchmark_end - benchmark_start}")
    print(f"{'='*70}\n")

    # Close the log file
    dual_writer.log_file.close()
    sys.stdout = dual_writer.console


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Redshift Serverless workgroup creation/deletion."
    )
    parser.add_argument(
        "--workgroup_name",
        type=str,
        default="benchmark",
        help="Base name for workgroups (RPU and iteration appended).",
    )
    parser.add_argument(
        "--iterations_per_rpu",
        type=int,
        default=5,
        help="Number of create/delete iterations per RPU size.",
    )
    parser.add_argument(
        "--min_rpu",
        type=int,
        default=4,
        help="Minimum RPU (power of 2).",
    )
    parser.add_argument(
        "--max_rpu",
        type=int,
        default=256,
        help="Maximum RPU (power of 2).",
    )
    args = parser.parse_args()

    benchmark_workgroup_creation(
        base_workgroup_name=args.workgroup_name,
        iterations_per_rpu=args.iterations_per_rpu,
        min_rpu=args.min_rpu,
        max_rpu=args.max_rpu,
    )
