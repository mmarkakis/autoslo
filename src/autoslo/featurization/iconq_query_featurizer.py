"""
Some code in this file was derived from code written by Ziniu Wu for IconqSched.
"""

import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, TypeAlias, cast

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace


class IconqQueryFeaturizer:
    """
    A class that featurizes a SQL query into a vectorized representation, using
    the top M operators in the query plans of a "training" set of queries, as
    well as the N largest tables in the database to define the feature space.
    """

    IconqQueryFeaturization: TypeAlias = list[float]
    """Represents the vectorized features of a query."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        schema_name: str,
        run_ids: list[str],
        m: int = 15,
        n: int = 20,
        use_size: bool = True,
        use_true_card: bool = False,
        use_table_selectivity: bool = False,
        use_log: bool = True,
        precomputed_top_operators: Optional[list[str]] = None,
        precomputed_top_tables: Optional[list[tuple[str, int]]] = None,
        precomputed_featurization_cache: Optional[
            dict[
                Trace.TPCDSTempAndQIdx,
                IconqQueryFeaturization,
            ]
        ] = None,
        from_sys_query_explain: bool = False,
    ) -> None:
        """
        Initializes the IconqQueryFeaturizer.

        Parameters:
            schema_name: The name of the schema containing the database tables.
            run_ids: The run IDs, the traces of which will be used to determine
                the top M operators and top N tables.
            m: The number of operators to consider.
            n: The number of tables to consider.
            use_size: Whether to multiply cardinalities by the width of
                operators.
            use_true_card: Whether to use the actual cardinality of operators,
                when available.
            use_table_selectivity: Whether to use the selectivity of tables as a
                feature, as opposed to the cardinality.
            use_log: Whether to use the logarithm of the cardinalities in the
                features.
            precomputed_top_operators: If provided, the top M operators to use.
            precomputed_top_tables: If provided, the top N tables to use.
            precomputed_featurization_cache: If provided, a cache mapping
                TPC-DS template and query indices to their featurizations.
            from_sys_query_explain: Whether the featurizer should operate based
                on sys_query_explain data (newer version), instead of parsed
                query plans (older version).
        """
        self._schema_name = schema_name
        self._run_ids = run_ids

        self._m = m
        self._top_operators: list[str] = []
        self._n = n
        self._top_tables: list[tuple[str, int]] = []  # (table_name, num rows)
        self._use_size = use_size
        self._use_true_card = use_true_card
        self._use_table_selectivity = use_table_selectivity
        self._use_log = use_log
        self._from_sys_query_explain = from_sys_query_explain

        self._featurization_cache: dict[
            Trace.TPCDSTempAndQIdx,
            IconqQueryFeaturizer.IconqQueryFeaturization,
        ] = {}

        if precomputed_top_operators is not None:
            self._top_operators = precomputed_top_operators
        else:
            print("Finding top operators...")
            self._top_operators = self._find_top_operators(run_ids)

        if precomputed_top_tables is not None:
            self._top_tables = precomputed_top_tables
        else:
            print("Finding top tables...")
            self._top_tables = self._find_top_tables()

        # Featurize all the given runs while we're at it, as a cache.
        if precomputed_featurization_cache is not None:
            self._featurization_cache = precomputed_featurization_cache
        else:
            print("Featurizing queries...")

            feat_func = (
                self.featurize_plan
                if not self._from_sys_query_explain
                else self.featurize_plan_from_sys_query_explain_rows
            )

            for run_id in tqdm(run_ids):
                trace = Trace(run_id)
                tpcds_temp_and_q_idxs = trace.tpcds_temp_and_q_idxs
                was_aborted = trace.was_aborted()

                info = (
                    trace.query_plans()
                    if not self._from_sys_query_explain
                    else trace.sys_query_explain_rows_per_query()
                )

                for query_id, aborted in was_aborted.items():
                    if aborted:
                        # Ignore aborted queries for accurate featurization.
                        continue

                    query_id = cast(str, query_id)
                    tpcds_temp_and_q_idx = tpcds_temp_and_q_idxs[query_id]

                    if tpcds_temp_and_q_idx in self._featurization_cache:
                        continue
                    if (query_id not in info) or (info[query_id] is None):
                        self._featurization_cache[tpcds_temp_and_q_idx] = []
                        continue
                    featurization = feat_func(info[query_id])  # type: ignore
                    self._featurization_cache[tpcds_temp_and_q_idx] = (
                        featurization
                    )

    @property
    def num_dims(self) -> int:
        """
        Returns the number of dimensions in the featurization vector.

        There are (2 * m) + n dimensions in the feature vector:
        - The first 2*m dimensions represent the top m operators in the query
            plan. Each of the m operators gets 2 features: one that counts the
            number of times the operator appears in the query plan, and one that
            counts the total number of rows estimated to be processed by the
            operator, within this plan.
        - The next n dimensions represent the top n tables in the database. For
            each of these tables, the corresponding feature measures the
            estimated cardinality of the data the current query reads from that
            table.

        Returns:
            The number of dimensions in the featurization vector.
        """
        return (2 * self._m) + self._n

    @property
    def top_table_names(self) -> list[str]:
        """
        Returns the names of the top N tables in the database.

        Returns:
            The names of the top N tables in the database.
        """
        return [table_name for (table_name, _) in self._top_tables]

    @property
    def run_ids(self) -> list[str]:
        """
        Returns the run IDs used to train this featurizer.

        Returns:
            The run IDs used to train this featurizer.
        """
        return self._run_ids

    def _find_top_operators(self, run_ids: list[str]) -> list[str]:
        """
        Finds the top M operators across the given runs. Based on the setting
        of self._from_sys_query_explain, either uses parsed query plans, or
        sys_query_explain data.

        Parameters:
            run_ids: The run IDs to use to find the top M operators.

        Returns:
            The top M operator names.
        """
        # Find all of the operators across queries and their counts.
        all_operators: dict[str, int] = defaultdict(int)

        if self._from_sys_query_explain:
            print("`from_sys_query_explain` is True; using sys_query_explain.")
            for run_id in tqdm(run_ids):
                trace = Trace(run_id)
                sys_query_explain_rows_per_query = (
                    trace.sys_query_explain_rows_per_query()
                )
                was_aborted = trace.was_aborted()
                for query_id, aborted in was_aborted.items():
                    if aborted:
                        # Ignore aborted queries for accurate operator counts.
                        continue
                    query_id = cast(str, query_id)  # Make mypy happy
                    sub_df = sys_query_explain_rows_per_query[query_id]
                    for node in sub_df["plan_node"].values:
                        operator_name = (
                            self._process_one_sys_query_explain_plan_node(node)[
                                0
                            ]
                        )
                        all_operators[operator_name] += 1
        else:
            print("`from_sys_query_explain` is False; using query plans.")
            query_plans = {}
            for run_id in tqdm(run_ids):
                trace = Trace(run_id)
                plans = trace.query_plans()
                query_plans.update(plans)
            for plan in tqdm(query_plans.values()):
                if plan is not None:
                    IconqQueryFeaturizer._dfs_count_operators(
                        plan, all_operators
                    )
        op_names = list(all_operators.keys())
        op_counts = list(all_operators.values())
        total_ops = sum(op_counts)

        # Sort the operators by frequency and return the top M.
        idx = np.argsort(op_counts)[::-1]
        m = min(self._m, len(op_names))
        explained_ops = sum(op_counts[i] for i in idx[:m])
        top_m = []
        for i in idx[:m]:
            top_m.append(op_names[i])
        print(
            f"Top {m} operators cover "
            f"{explained_ops}/{total_ops} = "
            f"{explained_ops/total_ops:.2%} of all operator occurrences."
        )
        print("Top operators:")
        for i, op in enumerate(top_m):
            print(f"  {i}: {op}")
        return top_m

    @staticmethod
    def _dfs_count_operators(plan: dict, all_operators: dict) -> None:
        """
        Updates an operator incidence dictionary by recursively traversing a
        query plan and counting the number of times each operator appears.

        Parameters:
            plan: The query plan to traverse.
            all_operators: A dictionary mapping operator names to their counts.
        """

        if "plan_parameters" in plan and "op_name" in plan["plan_parameters"]:
            op_name = plan["plan_parameters"]["op_name"]
            # Remove any table mentions from the operator name.
            # FIXME: remove any word including tpcds1000
            op_name = " ".join(
                [word for word in op_name.split() if "tpcds" not in word]
            )
            all_operators[op_name] = all_operators.get(op_name, 0) + 1
        if "children" in plan:
            for child in plan["children"]:
                IconqQueryFeaturizer._dfs_count_operators(child, all_operators)

    def _find_top_tables(self) -> list[tuple[str, int]]:
        """
        Finds the top N tables across the given query plans.

        Returns:
            A dictionary mapping table IDs to tuples of (table name, size).
        """

        # Load the database statistics.
        # FIXME: We just load from cluster size 32.
        statistics_path = os.path.join(
            pu.get_data_path(),
            "db_stats",
            f"cluster_32_{self._schema_name}.yml",
        )
        with open(statistics_path, "r", encoding="utf-8") as f:
            stats = yaml.safe_load(f)

        # Get tables in descending order by size.
        table_names_and_sizes: list[tuple[str, int]] = [
            (v["table_name"], int(v["num_rows"]))
            for v in stats["table_stats"].values()
        ]
        table_names_and_sizes.sort(key=lambda x: (-x[1], x[0]))
        print(f"Top {self._n} tables by size:")
        for table_name, size in table_names_and_sizes[: self._n]:
            print(f"  {table_name}: {size} rows")
        return table_names_and_sizes[: self._n]

    def featurize_from_tpcds_temp_and_q_idx(
        self,
        tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
    ) -> IconqQueryFeaturization:
        """
        Converts the query represented by the given TPC-DS query ID into a
        vectorized representation.

        Parameters:
            tpcds_temp_and_q_idx: The TPC-DS template and query index
                identifying the query to convert.

        Returns:
            The vectorized representation of the query text.

        Raises:
            ValueError: If the TPC-DS query ID is not in the cache.
        """

        if tpcds_temp_and_q_idx in self._featurization_cache:
            return self._featurization_cache[tpcds_temp_and_q_idx]

        raise ValueError("No cached featurization for this query.")

    def featurize(
        self,
        query_text: str,
    ) -> IconqQueryFeaturization:
        """
        Converts the given query text into a vectorized representation.

        Parameters:
            query_text: The text of the query to convert.

        Returns:
            The vectorized representation of the query text.

        Raises:
            ValueError: If the TPC-DS query ID cannot be extracted from the
                query text, or if there is no cached featurization for the
                extracted TPC-DS query ID.
        """

        tpcds_temp_and_q_idx = Trace.extract_temp_and_q_idxs(query_text)
        if tpcds_temp_and_q_idx is None:
            raise ValueError(
                "Could not extract TPC-DS template and query index."
            )

        return self.featurize_from_tpcds_temp_and_q_idx(tpcds_temp_and_q_idx)

    def featurize_trace(
        self, trace: Trace
    ) -> dict[str, IconqQueryFeaturization]:
        """
        Featurizes all queries in the given trace.

        Parameters:
            trace: The Trace object containing the queries to featurize.

        Returns:
            A dictionary mapping query IDs to their vectorized representations.
        """
        featurizations: dict[
            str, IconqQueryFeaturizer.IconqQueryFeaturization
        ] = {}
        tpcds_temp_and_q_idxs = trace.tpcds_temp_and_q_idxs

        for query_id, tpcds_temp_and_q_idx in tpcds_temp_and_q_idxs.items():
            query_id = cast(str, query_id)
            featurization = self.featurize_from_tpcds_temp_and_q_idx(
                tpcds_temp_and_q_idx
            )
            featurizations[query_id] = featurization

        return featurizations

    def dump_featurization(
        self, query_text: str, out_dir: Optional[str] = None
    ) -> None:
        """
        Dumps the featurization of the given query text to a file.

        Parameters:
            query_text: The text of the query to dump.
            out_dir: The directory to write the featurization to. If None, the
                featurization is written to a file in the current directory.
        """
        d: dict[str, Any] = {}
        d["query_text"] = query_text
        d["tpcds_temp_and_q_idx"] = Trace.extract_temp_and_q_idxs(query_text)
        d["featurization"] = self.featurize_from_tpcds_temp_and_q_idx(
            d["tpcds_temp_and_q_idx"]
        )

        d["human_readable_featurization"] = []
        for i, feature in enumerate(d["featurization"]):
            if i < 2 * self._m and i % 2 == 0:
                op_idx = i // 2
                operator_name = (
                    self._top_operators[op_idx]
                    if op_idx < len(self._top_operators)
                    else "None"
                )
                d["human_readable_featurization"].append(
                    {
                        "type": "operator_count",
                        "operator": operator_name,
                        "value": feature,
                    }
                )
            elif i < 2 * self._m and i % 2 == 1:
                op_idx = i // 2
                operator_name = (
                    self._top_operators[op_idx]
                    if op_idx < len(self._top_operators)
                    else "None"
                )
                d["human_readable_featurization"].append(
                    {
                        "type": "operator_cardinality",
                        "operator": operator_name,
                        "value": feature,
                    }
                )
            else:
                d["human_readable_featurization"].append(
                    {
                        "type": "table_cardinality",
                        "table": self.top_table_names[i - 2 * self._m],
                        "value": feature,
                    }
                )

        if out_dir is None:
            out_dir = "."
        os.makedirs(out_dir, exist_ok=True)
        out_file_path = os.path.join(
            out_dir, f'featurization_{d["query_hash"]}.yml'
        )
        with open(out_file_path, "w") as f:
            yaml.dump(d, f, sort_keys=False)

    def featurize_plan(
        self,
        query_plan: dict,
    ) -> IconqQueryFeaturization:
        """
        Converts the given query plan into a vectorized representation.

        Parameters:
            query_plan: The query plan to convert.

        Returns:
            The vectorized representation of the query plan.

        """
        # Compute the features of the query plan.
        features = [0.0] * self.num_dims
        self._dfs_compute_features(
            query_plan,
            features,
        )

        # Transform cardinality features, if needed.
        features = self._transform_card_if_needed(features)
        return features

    def _transform_card_if_needed(
        self, features: IconqQueryFeaturization
    ) -> IconqQueryFeaturization:
        """
        Transforms the cardinality features in the given feature vector, if
        needed.

        Parameters:
            features: The feature vector to transform.

        Returns:
            The transformed feature vector.
        """

        # For each feature representing an operator cardinality, either
        # take the log or convert to MB.
        for i in range(2 * self._m):
            # N.B. The original code also took a log of the operator count, if
            # self._use_log is True
            # if ((i % 2) == 0) and self._use_log:
            #    features[i] = max(np.log(features[i] + 1e-5), 0)
            # elif (i % 2) == 1:
            if (i % 2) == 1:
                features[i] = self._transform_card(features[i], self._use_log)

        # For each feature representing a table cardinality, either convert it
        # to a relative cardinality using the table size, or keep it as is; in
        # the latter case, either take the log or convert to MB.
        for i in range(2 * self._m, self.num_dims):
            if self._use_table_selectivity:
                _, table_size = self._top_tables[i - 2 * self._m]
                features[i] = features[i] / table_size
            else:
                features[i] = self._transform_card(features[i], self._use_log)

        return list(features)

    def _transform_card(self, card: float, use_log: bool) -> float:
        """
        Transforms a cardinality value into a feature. If `use_log` is True, the
        logarithm of the cardinality is returned; otherwise, the cardinality is
        divided by 1024*1024 to convert it to MB.

        Parameters:
            card: The cardinality value to transform.
            use_log: Whether to use the logarithm transformation.

        Returns:
            The transformed cardinality value.
        """
        if use_log:
            return float(max(np.log(card + 1e-5), 0))

        return card / 1024.0 / 1024.0

    def _dfs_compute_features(
        self,
        plan: dict,
        features: IconqQueryFeaturization,
    ) -> None:
        """
        Recursively computes the features of a query plan.

        Parameters:
            plan: The query plan to traverse.
            features: The feature vector to update.
        """

        if "plan_parameters" in plan and "op_name" in plan["plan_parameters"]:
            op = plan["plan_parameters"]["op_name"]
            op = " ".join([word for word in op.split() if "tpcds" not in word])
            if op in self._top_operators:

                # Compute cardinality of the operator.
                if self._use_true_card and (
                    "act_card" in plan["plan_parameters"]
                ):
                    card = plan["plan_parameters"]["act_card"]
                else:
                    card = plan["plan_parameters"]["est_card"]
                if self._use_size:
                    card = card * plan["plan_parameters"]["est_width"]

                # Update operator-related features.
                idx = self._top_operators.index(op)
                features[idx * 2] += 1
                features[idx * 2 + 1] += card

                # Update table-related features.
                if "table" in plan["plan_parameters"]:
                    table_name = plan["plan_parameters"]["table"]
                    for i, (top_table_name, _) in enumerate(self._top_tables):
                        if table_name == top_table_name:
                            table_idx = 2 * self._m + i
                            features[table_idx] += card

        # Recursively traverse the children of the current operator.
        if "children" in plan:
            for child_plan in plan["children"]:
                self._dfs_compute_features(
                    child_plan,
                    features,
                )

    def save(self, timestamp:Optional[str] = None) -> str:
        """
        Saves the IconqQueryFeaturizer.

        Returns:
            The identifier of the saved IconqQueryFeaturizer. This is a
                subdirectory under `data/iconq_query_featurizations/` named
                with the current timestamp.
        """
        # Create directory.
        if timestamp is None:
            timestamp = str(int(datetime.now().timestamp()))
        save_dir = os.path.join(
            pu.get_data_path(),
            "iconq_query_featurizations",
            timestamp,
        )
        os.makedirs(save_dir, exist_ok=(timestamp is not None)) 

        # Save featurizer parameters.
        param_path = os.path.join(save_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.safe_dump(
                {
                    "m": self._m,
                    "n": self._n,
                    "use_size": self._use_size,
                    "use_true_card": self._use_true_card,
                    "use_table_selectivity": self._use_table_selectivity,
                    "use_log": self._use_log,
                    "schema_name": self._schema_name,
                    "run_ids": self._run_ids,
                    "top_operators": self._top_operators,
                    "top_tables": self._top_tables,
                },
                f,
            )

        # Save featurzation cache.
        cache_path = os.path.join(save_dir, "featurizations.yml")
        with open(cache_path, "w") as f:
            l = []
            for (
                tpcds_temp_and_q_idx,
                featurization,
            ) in self._featurization_cache.items():
                l.append(
                    {
                        "tpcds_temp_and_q_idx": tpcds_temp_and_q_idx,
                        "featurization": featurization,
                    }
                )
            yaml.safe_dump(l, f, sort_keys=False)

        # Also save it as a pickle for easier loading.
        pickle_path = os.path.join(save_dir, "featurizations.pkl")
        with open(pickle_path, "wb") as f:
            print('here')
            pickle.dump(self._featurization_cache, f)

        return timestamp

    @staticmethod
    def load(timestamp: str) -> "IconqQueryFeaturizer":
        """
        Loads a IconqQueryFeaturizer from a directory.

        Parameters:
            timestamp: The timestamp of the directory to load the
                IconqQueryFeaturizer from.
        """

        # Load parameters.
        load_dir = os.path.join(
            pu.get_data_path(),
            "iconq_query_featurizations",
            timestamp,
        )
        param_path = os.path.join(load_dir, "params.yml")
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        # Load featurization cache.
        cache_path = os.path.join(load_dir, "featurizations.pkl")
        # Temp fix: if pickle file doesn't exist, try loading from yaml file
        # and immediately dump to pickle for future use.
        if not os.path.exists(cache_path):
            print(
                f"Pickle file not found at {cache_path}, trying to load from yaml file."
            )
            yaml_cache_path = os.path.join(load_dir, "featurizations.yml")
            with open(yaml_cache_path, "r") as f:
                l = yaml.safe_load(f)
            precomputed_featurization_cache = {
                item["tpcds_temp_and_q_idx"]: item["featurization"]
                for item in l
            }
        else:
            with open(cache_path, "rb") as f:
                precomputed_featurization_cache = pickle.load(f)

        featurizer = IconqQueryFeaturizer(
            schema_name=params["schema_name"],
            run_ids=params["run_ids"],
            m=params["m"],
            n=params["n"],
            use_size=params["use_size"],
            use_true_card=params["use_true_card"],
            use_table_selectivity=params["use_table_selectivity"],
            use_log=params["use_log"],
            precomputed_top_operators=params["top_operators"],
            precomputed_top_tables=params["top_tables"],
            precomputed_featurization_cache=precomputed_featurization_cache,
        )

        if not os.path.exists(cache_path):
            featurizer.save(timestamp)  # Save to pickle for future use.

        return featurizer

    def table_access_pattern_cosine_similarity_from_tpcds_temp_and_q_idxs(
        self,
        tpcds_temp_and_q_idx_a: Trace.TPCDSTempAndQIdx,
        tpcds_temp_and_q_idx_b: Trace.TPCDSTempAndQIdx,
        binarize: float = False,
    ) -> float:
        """
        Computes the cosine similarity between two queries based on their table
        access patterns, given their TPC-DS template and query indices.

        Parameters:
            tpcds_temp_and_q_idx_a: The TPC-DS template and query index of the
                first query.
            tpcds_temp_and_q_idx_b: The TPC-DS template and query index of the
                second query.
            binarize: Whether to ignore the exact values of the table-related
                features, instead only considering whether or not they are zero.

        Returns:
            A cosine similarity score between 0 and 1, where 1 means identical
            table access patterns.
        """
        featurization_a = self.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx_a
        )
        featurization_b = self.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx_b
        )
        return self.table_access_pattern_cosine_similarity(
            featurization_a, featurization_b, binarize
        )

    def table_access_pattern_cosine_similarity(
        self,
        featurization_a: IconqQueryFeaturization,
        featurization_b: IconqQueryFeaturization,
        binarize: float = False,
    ) -> float:
        """
        Computes the cosine similarity between two queries based on their table
        access patterns.

        Parameters:
            featurization_a: The featurization of the first query.
            featurization_b: The featurization of the second query.
            binarize: Whether to ignore the exact values of the table-related
                features, instead only considering whether or not they are zero.

        Returns:
            A cosine similarity score between 0 and 1, where 1 means identical
            table access patterns.
        """
        table_features_a = featurization_a[2 * self._m :]
        table_features_b = featurization_b[2 * self._m :]

        if binarize:
            table_features_a = [int(f != 0) for f in table_features_a]
            table_features_b = [int(f != 0) for f in table_features_b]

        dot_product = sum(
            a * b for a, b in zip(table_features_a, table_features_b)
        )

        norm_a = sum(a * a for a in table_features_a) ** 0.5
        norm_b = sum(b * b for b in table_features_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def table_access_pattern_coverage_from_tpcds_temp_and_q_idxs(
        self,
        tpcds_temp_and_q_idx_a: Trace.TPCDSTempAndQIdx,
        tpcds_temp_and_q_idx_b: Trace.TPCDSTempAndQIdx,
    ) -> float:
        """
        Computes the coverage between two queries based on their table
        access patterns, given their TPC-DS template and query indices. Assume
        that query A is the reference query, and query B is the query whose
        coverage is being measured.

        Parameters:
            tpcds_temp_and_q_idx_a: The TPC-DS template and query index of the
                first query.
            tpcds_temp_and_q_idx_b: The TPC-DS template and query index of the
                second query.

        Returns:
            A coverage score between 0 and 1, where 1 means query A had already
            accessed all tables that query B accessed.
        """
        featurization_a = self.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx_a
        )
        featurization_b = self.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx_b
        )
        return self.table_access_pattern_coverage(
            featurization_a, featurization_b
        )

    def table_access_pattern_coverage(
        self,
        featurization_a: IconqQueryFeaturization,
        featurization_b: IconqQueryFeaturization,
    ) -> float:
        """
        Computes the coverage between two queries based on their table
        access patterns. Assume that query A is the reference query, and query B
        is the query whose coverage is being measured.

        Parameters:
            featurization_a: The featurization of the first query.
            featurization_b: The featurization of the second query.

        Returns:
            A coverage score between 0 and 1, where 1 means query A had already
            accessed all tables that query B accessed.
        """
        table_features_a = featurization_a[2 * self._m :]
        table_features_b = featurization_b[2 * self._m :]

        numerator = sum(
            [min(a, b) for a, b in zip(table_features_a, table_features_b)]
        )
        denominator = sum(table_features_b)

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def nonzero_feature_for_table_from_tpcds_temp_and_q_idx(
        self, tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx, table_name: str
    ) -> bool:
        """
        Answers whether the featurization for the given query has a non-zero
        value in the feature corresponding to the given table. Useful for
        sanity checking.
        """

        featurization = self.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx
        )
        return self.nonzero_feature_for_table(featurization, table_name)

    def nonzero_feature_for_table(
        self, featurization: IconqQueryFeaturization, table_name: str
    ) -> bool:
        """
        Answers whether the featurization for the given query has a non-zero
        value in the feature corresponding to the given table. Useful for
        sanity checking.
        """

        if table_name not in self.top_table_names:
            raise ValueError(f"Table {table_name} is not in the top tables.")
        table_idx = self.top_table_names.index(table_name)
        feature_idx = 2 * self._m + table_idx
        return featurization[feature_idx] > 0.0

    def _process_one_sys_query_explain_plan_node(
        self, plan_node: str
    ) -> tuple[str, Optional[str], float, float]:
        """
        Processes one plan node from sys_query_explain.

        Parameters:
            plan_node: The plan node string.

        Returns:
            operator_name: The name of the operator.
            base_table_name: The name of the base table, if any.
            cardinality: The estimated cardinality of the operator.
            width: The estimated width of the operator.
        """

        # -> XN Seq Scan ext_tpcds1000.date_dim d2 (cost=0.00..1095.73 rows=30 width=4)
        operator_name = (
            plan_node.strip().lstrip("->").strip().split("(")[0].strip()
        )
        base_table_name = None
        if "Seq Scan" in operator_name:
            base_table_name = (
                operator_name.split("Seq Scan")[-1]
                .strip()
                .split()[0]
                .split(".")[-1]
                .strip()
            )
        if "Scan" in operator_name:
            operator_name = operator_name.split("Scan")[0] + "Scan"

        cardinality = plan_node.split("rows=")[1].split(" ")[0].strip()
        width = plan_node.split("width=")[1].split(")")[0].strip()

        return operator_name, base_table_name, float(cardinality), float(width)

    def featurize_plan_from_sys_query_explain_rows(
        self,
        sys_query_explain_sub_df: pd.DataFrame,
        child_queries_to_ignore: Optional[set[int]] = None,
    ) -> IconqQueryFeaturization:
        """
        Converts the given sys_query_explain rows into a vectorized representation.

        Parameters:
            sys_query_explain_sub_df: The sys_query_explain rows to convert.

        Returns:
            The vectorized representation of the query plan.

        Raises:
            ValueError: If the rows don't all correspond to the same query.
        """
        if len(sys_query_explain_sub_df["query_id"].unique()) != 1:
            raise ValueError(
                "The provided sys_query_explain rows do not all correspond to "
                "the same query."
            )

        features = [0.0] * self.num_dims

        if child_queries_to_ignore is None:
            child_queries_to_ignore = set()

        for _, row in sys_query_explain_sub_df.iterrows():
            child_query_id = row["child_query_sequence"]
            if child_query_id in child_queries_to_ignore:
                continue
            plan_node: str = row["plan_node"]
            operator_name, base_table_name, cardinality, width = (
                self._process_one_sys_query_explain_plan_node(plan_node)
            )

            if self._use_size:
                cardinality = float(cardinality) * max(1.0, float(width))

            # Update operator-related features.
            if operator_name in self._top_operators:
                idx = self._top_operators.index(operator_name)
                # Update operator-related features.
                features_idx_count = idx * 2
                features_idx_card = idx * 2 + 1
                features[features_idx_count] += 1
                features[features_idx_card] += float(cardinality)

            # Update table-related features.
            if (base_table_name is not None) and (
                base_table_name in self.top_table_names
            ):
                table_idx = self.top_table_names.index(base_table_name)
                feature_idx = 2 * self._m + table_idx
                features[feature_idx] += float(cardinality)

        # Transform cardinality features, if needed.
        features = self._transform_card_if_needed(features)
        return features
