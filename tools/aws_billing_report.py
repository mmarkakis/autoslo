#!/usr/bin/env python3

from collections import defaultdict
from datetime import date, timedelta

import boto3
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def main():
    console = Console()

    # Cost Explorer expects an inclusive start date and an exclusive end date.
    today = date.today()
    start = today.replace(day=1)
    end = today + timedelta(days=1)

    # Use the default AWS credential chain:
    # ~/.aws/credentials, AWS_PROFILE, env vars, IAM role, etc.
    ce = boto3.client("ce")

    # Fetch current-month costs grouped by AWS service.
    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start.isoformat(),
            "End": end.isoformat(),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[
            {"Type": "DIMENSION", "Key": "SERVICE"},
        ],
    )

    # Aggregate costs by service.
    costs = defaultdict(float)
    currency = "USD"

    for result in response["ResultsByTime"]:
        for group in result["Groups"]:
            service = group["Keys"][0]
            amount_info = group["Metrics"]["UnblendedCost"]

            costs[service] += float(amount_info["Amount"])
            currency = amount_info["Unit"]

    total = sum(costs.values())

    # Print summary panel.
    console.print(
        Panel.fit(
            f"[bold]Period:[/bold] {start.isoformat()} to {end.isoformat()} "
            f"(exclusive)\n"
            f"[bold]Total:[/bold] {total:,.2f} {currency}",
            title="Summary",
        )
    )

    # Print per-service breakdown.
    table = Table(title="AWS Billing Report — Current Month")
    table.add_column("Service", style="bold")
    table.add_column("Cost", justify="right")
    table.add_column("Share", justify="right")

    for service, cost in sorted(
        costs.items(), key=lambda x: x[1], reverse=True
    ):
        if cost <= 0:
            continue

        share = (cost / total * 100) if total else 0.0

        table.add_row(
            service,
            f"{cost:,.2f} {currency}",
            f"{share:.1f}%",
        )

    console.print(table)


if __name__ == "__main__":
    main()
