#!/usr/bin/env python3
"""Delete all Redshift Serverless workgroups starting with 'autoslo' and their namespaces."""

import time

import boto3
from botocore.exceptions import ClientError
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text

PREFIX = "autoslo"
POLL_INTERVAL_S = 5
AVAILABLE_TIMEOUT_S = 120

console = Console()


def _list_workgroups(client) -> list[dict]:
    results = []
    paginator = client.get_paginator("list_workgroups")
    for page in paginator.paginate():
        for wg in page["workgroups"]:
            if wg["workgroupName"].startswith(PREFIX):
                results.append(wg)
    return results


def _list_namespaces(client) -> list[dict]:
    results = []
    paginator = client.get_paginator("list_namespaces")
    for page in paginator.paginate():
        for ns in page["namespaces"]:
            if ns["namespaceName"].startswith(PREFIX):
                results.append(ns)
    return results


def _status_style(status: str) -> str:
    s = status.upper()
    if s == "AVAILABLE":
        return "green"
    if s == "DELETING":
        return "yellow"
    if s in ("CREATING", "MODIFYING"):
        return "cyan"
    return "red"


def _build_workgroup_table(workgroups: list[dict]) -> Table:
    table = Table(title="Workgroups", show_lines=True)
    table.add_column("Workgroup", style="bold")
    table.add_column("Namespace")
    table.add_column("Status")
    table.add_column("Base RPU", justify="right")
    for wg in workgroups:
        table.add_row(
            wg["workgroupName"],
            wg.get("namespaceName", ""),
            Text(wg["status"], style=_status_style(wg["status"])),
            str(wg.get("baseCapacity", "")),
        )
    return table


def _build_namespace_table(namespaces: list[dict]) -> Table:
    table = Table(title="Namespaces", show_lines=True)
    table.add_column("Namespace", style="bold")
    table.add_column("Status")
    for ns in namespaces:
        table.add_row(
            ns["namespaceName"],
            Text(ns["status"], style=_status_style(ns["status"])),
        )
    return table


def _wait_for_workgroup_deletion(client, names: list[str]) -> None:
    remaining = set(names)
    with Live(console=console, refresh_per_second=2) as live:
        while remaining:
            still_present = set()
            for name in remaining:
                try:
                    resp = client.get_workgroup(workgroupName=name)
                    status = resp["workgroup"]["status"]
                    still_present.add(name)
                except ClientError as e:
                    if (
                        e.response["Error"]["Code"]
                        == "ResourceNotFoundException"
                    ):
                        console.log(
                            f"[green]Workgroup '{name}' deleted.[/green]"
                        )
                    else:
                        raise
            remaining = still_present
            if remaining:
                items = ", ".join(sorted(remaining))
                live.update(
                    Text(
                        f"Waiting for workgroups to delete: {items} ...",
                        style="yellow",
                    )
                )
                time.sleep(POLL_INTERVAL_S)


def _wait_for_namespace_deletion(client, names: list[str]) -> None:
    remaining = set(names)
    with Live(console=console, refresh_per_second=2) as live:
        while remaining:
            still_present = set()
            for name in remaining:
                try:
                    resp = client.get_namespace(namespaceName=name)
                    status = resp["namespace"]["status"]
                    still_present.add(name)
                except ClientError as e:
                    if (
                        e.response["Error"]["Code"]
                        == "ResourceNotFoundException"
                    ):
                        console.log(
                            f"[green]Namespace '{name}' deleted.[/green]"
                        )
                    else:
                        raise
            remaining = still_present
            if remaining:
                items = ", ".join(sorted(remaining))
                live.update(
                    Text(
                        f"Waiting for namespaces to delete: {items} ...",
                        style="yellow",
                    )
                )
                time.sleep(POLL_INTERVAL_S)


def _wait_until_workgroup_available(client, name: str) -> bool:
    """Block until the workgroup reaches AVAILABLE status.

    Returns True if AVAILABLE, False if timed out.
    """
    deadline = time.monotonic() + AVAILABLE_TIMEOUT_S
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            resp = client.get_workgroup(workgroupName=name)
            status = resp["workgroup"]["status"]
            if status == "AVAILABLE":
                return True
            if time.monotonic() >= deadline:
                console.log(
                    f"[red]Timed out waiting for workgroup '{name}' to become "
                    f"AVAILABLE (stuck at {status}) after {AVAILABLE_TIMEOUT_S}s.[/red]"
                )
                return False
            elapsed = AVAILABLE_TIMEOUT_S - (deadline - time.monotonic())
            live.update(
                Text(
                    f"Waiting for workgroup '{name}' to become AVAILABLE "
                    f"(currently {status}, {elapsed:.0f}/{AVAILABLE_TIMEOUT_S}s) ...",
                    style="yellow",
                )
            )
            time.sleep(POLL_INTERVAL_S)


def _delete_workgroup_with_retry(client, name: str) -> None:
    """Request workgroup deletion, retrying on ConflictException."""
    while True:
        try:
            client.delete_workgroup(workgroupName=name)
            return
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConflictException":
                console.log(
                    f"[yellow]ConflictException for '{name}', retrying in {POLL_INTERVAL_S}s ...[/yellow]"
                )
                time.sleep(POLL_INTERVAL_S)
            else:
                raise


def main() -> None:
    client = boto3.client("redshift-serverless")

    workgroups = _list_workgroups(client)
    namespaces = _list_namespaces(client)

    if not workgroups and not namespaces:
        console.print(
            f"[green]No workgroups or namespaces found with prefix '{PREFIX}'.[/green]"
        )
        return

    if workgroups:
        console.print()
        console.print(_build_workgroup_table(workgroups))
    if namespaces:
        console.print()
        console.print(_build_namespace_table(namespaces))

    wg_count = len(workgroups)
    ns_count = len(namespaces)
    console.print(
        f"\n[bold red]This will delete {wg_count} workgroup(s) and {ns_count} namespace(s) "
        f"WITHOUT final snapshots.[/bold red]"
    )
    # Partition namespaces into those backed by a workgroup vs orphaned.
    wg_namespace_names: set[str] = {
        wg.get("namespaceName", "") for wg in workgroups
    }
    orphan_namespaces = [
        ns for ns in namespaces if ns["namespaceName"] not in wg_namespace_names
    ]
    backed_namespaces = [
        ns for ns in namespaces if ns["namespaceName"] in wg_namespace_names
    ]

    if orphan_namespaces:
        console.print(
            f"\n  [dim]{len(orphan_namespaces)} namespace(s) have no associated workgroup (orphaned).[/dim]"
        )

    answer = console.input("[bold]Proceed? [y/N] [/bold]").strip().lower()
    if answer != "y":
        console.print("[yellow]Aborted.[/yellow]")
        return

    # --- Delete orphaned namespaces first (no workgroup to wait for) ---
    orphan_ns_names = []
    for ns in orphan_namespaces:
        name = ns["namespaceName"]
        console.log(
            f"Requesting deletion of orphaned namespace [bold]{name}[/bold] (no snapshot) ..."
        )
        client.delete_namespace(namespaceName=name, finalSnapshotName="")
        orphan_ns_names.append(name)

    # --- Delete workgroups ---
    wg_names = []
    skipped_wg_ns: set[str] = set()
    for wg in workgroups:
        name = wg["workgroupName"]
        if not _wait_until_workgroup_available(client, name):
            console.log(
                f"[red]Skipping workgroup '{name}' (not available).[/red]"
            )
            ns = wg.get("namespaceName", "")
            if ns:
                skipped_wg_ns.add(ns)
            continue
        console.log(f"Requesting deletion of workgroup [bold]{name}[/bold] ...")
        _delete_workgroup_with_retry(client, name)
        wg_names.append(name)

    if wg_names:
        _wait_for_workgroup_deletion(client, wg_names)

    # --- Delete workgroup-backed namespaces (skip those whose workgroup was skipped) ---
    backed_ns_names = []
    for ns in backed_namespaces:
        name = ns["namespaceName"]
        if name in skipped_wg_ns:
            console.log(
                f"[red]Skipping namespace '{name}' (its workgroup was not deleted).[/red]"
            )
            continue
        console.log(
            f"Requesting deletion of namespace [bold]{name}[/bold] (no snapshot) ..."
        )
        client.delete_namespace(namespaceName=name, finalSnapshotName="")
        backed_ns_names.append(name)

    # --- Wait for all namespace deletions ---
    all_ns_names = orphan_ns_names + backed_ns_names
    if all_ns_names:
        _wait_for_namespace_deletion(client, all_ns_names)

    if skipped_wg_ns:
        console.print(
            f"\n[bold red]Warning: {len(skipped_wg_ns)} namespace(s) were NOT deleted "
            f"because their workgroups could not be deleted: {', '.join(sorted(skipped_wg_ns))}[/bold red]"
        )

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
