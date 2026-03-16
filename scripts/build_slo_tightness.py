#!/usr/bin/env python3
"""Build a per-template SLO-tightness YAML from a StageModel and SloResolver.

Usage::

    python scripts/build_slo_tightness.py \\
        --stage-model-id 1771539369 \\
        --slo-file ext_tpcds1000_rpu16_p50_k1.5.yml \\
        --default-slo 30.0 \\
        --schema-name ext_tpcds1000 \\
        --reference-rpu 16 \\
        --template-ids 1 2 3 14 42 67 78 83 95 99 \\
        --output data/slo_tightness/tpcds1000_rpu16.yml

For each template the script computes::

    tightness = isolated_prediction / slo

Values > 1 mean the query already exceeds its SLO in isolation.
"""

from __future__ import annotations

import argparse
import os

import yaml

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.stage_model import StageModel
from autoslo.workload_definition.query import QueryTextId


def _compute_tightness(
    stage_model: StageModel,
    slo_resolver: SloResolver,
    schema_name: str,
    reference_rpu: int,
    template_ids: list[str],
) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for tid in template_ids:
        # Use query index 000 as representative.
        qtid = QueryTextId(value=f"{schema_name}#{tid}#000")
        dummy_qid = f"__tightness_{tid}"

        preds = stage_model.predict_from_query_text_id(
            {dummy_qid: qtid}, cluster_rpu=reference_rpu
        )
        iso_pred_s = preds[dummy_qid].overall_mean_s()

        slo_s = slo_resolver.resolve(qtid)

        if slo_s <= 0:
            tightness = float('inf')
        else:
            tightness = iso_pred_s / slo_s

        entries[str(tid)] = {
            "isolated_prediction_s": round(iso_pred_s, 4),
            "slo_s": round(slo_s, 4),
            "tightness": round(tightness, 4),
        }

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-template SLO-tightness YAML."
    )
    parser.add_argument(
        "--stage-model-id",
        required=True,
        help="Timestamp / ID of the StageModel to load.",
    )
    parser.add_argument(
        "--slo-file",
        default=None,
        help="SLO override filename under data/slos/ (optional).",
    )
    parser.add_argument(
        "--default-slo",
        type=float,
        default=30.0,
        help="Default SLO in seconds (used when no override exists).",
    )
    parser.add_argument(
        "--schema-name",
        required=True,
        help="Schema name (e.g. ext_tpcds1000).",
    )
    parser.add_argument(
        "--reference-rpu",
        type=int,
        required=True,
        help="RPU size for isolated predictions.",
    )
    parser.add_argument(
        "--template-ids",
        nargs="+",
        required=True,
        help="Template IDs to include.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YAML path (e.g. data/slo_tightness/my_table.yml).",
    )
    args = parser.parse_args()

    stage_model = StageModel.load(args.stage_model_id)
    slo_resolver = SloResolver(
        default_slo_s=args.default_slo,
        slo_dict_filename=args.slo_file,
    )

    entries = _compute_tightness(
        stage_model=stage_model,
        slo_resolver=slo_resolver,
        schema_name=args.schema_name,
        reference_rpu=args.reference_rpu,
        template_ids=args.template_ids,
    )

    result = {
        "schema_name": args.schema_name,
        "reference_rpu": args.reference_rpu,
        "stage_model_id": args.stage_model_id,
        "slo_source": args.slo_file or f"default_{args.default_slo}s",
        "entries": entries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(result, f, sort_keys=False, default_flow_style=False)

    print(
        f"Wrote {len(entries)} entries to {args.output} "
        f"(reference RPU={args.reference_rpu})"
    )


if __name__ == "__main__":
    main()
