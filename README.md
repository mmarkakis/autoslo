# AutoSLO: Practical Latency SLOs on Cloud Data Warehouses

We present AutoSLO, a **latency-SLO-aware** workload management framework for
**multi-cluster cloud data warehouses**. AutoSLO operates across three timescales through
three key components. 

The **Policy Tuner** can plan using the historical workload. It generates and simulates
forecasted workloads to determine proactive cluster-management actions and configure
parameters. By ensuring that the right resources are available at the right time
(shortly before demand materializes), it reduces dependence on reactive scaling and
mitigates cluster spinup delays.

The **Autoscaler** can adjust the active cluster set when observed workload behavior
deviates from the forecast. Its important contributions include cluster
spinup/teardown triggers, which identify opportune moments for resource adjustments,
and a cluster spinup size selector, which uses short-term simulations to estimate
the SLO and cost implications of different scaling actions.

Finally, the **Query Router** can react to live load. When a query *q* arrives, it
selects on which active cluster it should be executed by managing a multi-way
tradeoff among (1) the risk that *q* will violate its SLO; (2) the risk *q* poses
to the SLO adherence of already-running queries; and (3) the resource cost implied
by each routing decision.

- Full API reference and documentation: **[mmarkakis.github.io/autoslo](https://mmarkakis.github.io/autoslo/)**

- Detailed map of paper concepts to repository files: [Paper-to-code-mapping](#paper-to-code-mapping)

- Hands-on instructions on using our top-level scripts: 
[Entry point reference](https://mmarkakis.github.io/autoslo/ops/entry_points/).


---

## Installation

```bash
pip install -e .
```

To build the documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Cite This Work

```bibtex
@misc{markakis2026autoslo,
title={AutoSLO: Practical Latency SLOs on Cloud Data Warehouses -- Extended Version}, 
author={Markos Markakis and Tim Kraska},
year={2026},
eprint={2607.11770},
archivePrefix={arXiv},
primaryClass={cs.DB},
url={https://arxiv.org/abs/2607.11770},
doi={10.48550/arXiv.2607.11770}
}
```

A machine-readable citation is also available in [CITATION.cff](https://github.com/mmarkakis/autoslo/blob/main/CITATION.cff).
