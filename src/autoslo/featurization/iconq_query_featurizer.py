"""
Some code in this file was derived from code written by Ziniu Wu for IconqSched.
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Optional, TypeAlias, cast

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

import autoslo.filesystem.path_utils as pu
from autoslo.workload_definition.query import ClusterAwareQueryId, QueryTextId
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
            dict[QueryTextId, IconqQueryFeaturization]
        ] = None,
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
                query text IDs to their featurizations.
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

        self._featurization_cache: dict[
            QueryTextId, IconqQueryFeaturizer.IconqQueryFeaturization
        ] = {}
        # Lazy numpy cache: populated on first call to
        # featurize_from_query_text_id_as_numpy.
        self._np_featurization_cache: dict[QueryTextId, np.ndarray] = {}

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

            for run_id in tqdm(run_ids):
                trace = Trace(run_id)
                query_text_ids = trace.query_text_ids
                was_aborted = trace.was_aborted()

                explain_rows = trace.sys_query_explain_rows_per_query()

                for query_id, aborted in was_aborted.items():
                    if aborted:
                        # Ignore aborted queries for accurate featurization.
                        continue

                    query_id = cast(str, query_id)
                    query_text_id = query_text_ids[query_id]

                    if query_text_id in self._featurization_cache:
                        continue
                    if (query_id not in explain_rows) or (
                        explain_rows[query_id] is None
                    ):
                        self._featurization_cache[query_text_id] = []
                        continue
                    featurization = (
                        self.featurize_plan_from_sys_query_explain_rows(
                            explain_rows[query_id]
                        )
                    )
                    self._featurization_cache[query_text_id] = featurization

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
    def num_tables(self) -> int:
        """
        Returns the number of tables (N) that this featurizer considers.

        Returns:
            The number of tables (N) that this featurizer considers.
        """
        return self._n

    def _find_top_operators(self, run_ids: list[str]) -> list[str]:
        """
        Finds the top M operators across the given runs.

        Parameters:
            run_ids: The run IDs to use to find the top M operators.

        Returns:
            The top M operator names.
        """
        # Find all of the operators across queries and their counts.
        all_operators: dict[str, int] = defaultdict(int)

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
                        self._process_one_sys_query_explain_plan_node(node)[0]
                    )
                    all_operators[operator_name] += 1

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

    def featurize_from_query_text_id(
        self, query_text_id: QueryTextId | str
    ) -> IconqQueryFeaturization:
        """
        Converts the query identified by *query_text_id* into a vectorized
        representation.

        Parameters:
            query_text_id: The query text ID identifying the query to convert.

        Returns:
            The vectorized representation of the query.

        Raises:
            ValueError: If *query_text_id* is not in the featurization cache.
        """
        if isinstance(query_text_id, str):
            query_text_id = QueryTextId(query_text_id)

        if query_text_id in self._featurization_cache:
            return self._featurization_cache[query_text_id]

        raise ValueError(
            f"No cached featurization for query_text_id '{query_text_id}'."
        )

    def featurize_from_query_text_id_as_numpy(
        self,
        query_text_id: QueryTextId,
    ) -> np.ndarray:
        """
        Like :meth:`featurize_from_query_text_id`, but returns the featurization
        as a float32 numpy array. Results are cached so the list[float] →
        np.ndarray conversion happens at most once per distinct query text ID.

        Parameters:
            query_text_id: The query text ID identifying the query to convert.

        Returns:
            The featurization as a 1-D float32 numpy array of shape (num_dims,).
        """
        if query_text_id not in self._np_featurization_cache:
            self._np_featurization_cache[query_text_id] = np.array(
                self.featurize_from_query_text_id(query_text_id),
                dtype=np.float32,
            )
        return self._np_featurization_cache[query_text_id]

    def table_vector_for(self, query_text_id: QueryTextId) -> np.ndarray:
        """Return the table-access slice of the featurization for *query_text_id*.

        Slices off the trailing *N* dimensions (``feat[2*m:]``) and returns
        them as a float64 array of shape ``(N,)``.  The underlying numpy
        featurization cache is populated as a side-effect.

        Parameters:
            query_text_id: The query text ID to look up.

        Returns:
            A 1-D float64 numpy array of length ``self._n``.

        Raises:
            ValueError: If *query_text_id* is not in the featurization cache.
        """
        feat = self.featurize_from_query_text_id_as_numpy(query_text_id)
        return feat[2 * self._m :].astype(np.float64)

    def warm_up_cache(self, query_text_ids: list[QueryTextId]) -> None:
        """
        Pre-populate both _featurization_cache and _np_featurization_cache for
        the given query text IDs. This amortizes cold-cache costs when
        running multiple simulations with the same workload.

        Parameters:
            query_text_ids: A list of query text IDs to featurize and cache.
        """
        for query_text_id in tqdm(
            query_text_ids, desc="Warming up IconqQueryFeaturizer cache"
        ):
            # This call populates _featurization_cache.
            _ = self.featurize_from_query_text_id(query_text_id)
            # This call populates _np_featurization_cache.
            _ = self.featurize_from_query_text_id_as_numpy(query_text_id)

    def featurize_trace(
        self, trace: Trace
    ) -> dict[ClusterAwareQueryId, IconqQueryFeaturization]:
        """
        Featurizes all queries in the given trace.

        Parameters:
            trace: The Trace object containing the queries to featurize.

        Returns:
            A dictionary mapping cluster-aware query IDs to their vectorized
            representations.
        """
        featurizations: dict[
            ClusterAwareQueryId, IconqQueryFeaturizer.IconqQueryFeaturization
        ] = {}
        query_text_ids = trace.query_text_ids

        for cluster_aware_query_id, query_text_id in query_text_ids.items():
            cluster_aware_query_id = cast(
                ClusterAwareQueryId, cluster_aware_query_id
            )
            featurization = self.featurize_from_query_text_id(query_text_id)
            featurizations[cluster_aware_query_id] = featurization

        return featurizations

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

    def save(self, iconq_query_featurizer_id: Optional[str] = None) -> str:
        """
        Saves the IconqQueryFeaturizer.

        Returns:
            An iconq_query_featurizer_id that uniquely identifies the saved
            featurizer (within its schema).
        """
        # Create directory.
        if iconq_query_featurizer_id is None:
            iconq_query_featurizer_id = str(int(datetime.now().timestamp()))
        save_dir = os.path.join(
            pu.get_data_path(),
            "__query_featurizations",
            self._schema_name,
            iconq_query_featurizer_id,
        )
        os.makedirs(save_dir, exist_ok=(iconq_query_featurizer_id is not None))

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
                query_text_id,
                featurization,
            ) in self._featurization_cache.items():
                l.append(
                    {
                        "query_text_id": query_text_id,
                        "featurization": featurization,
                    }
                )
            yaml.safe_dump(l, f, sort_keys=False)

        return iconq_query_featurizer_id

    @staticmethod
    def load(
        schema_name: str, iconq_query_featurizer_id: str
    ) -> "IconqQueryFeaturizer":
        """
        Loads a IconqQueryFeaturizer from a directory.

        Parameters:
            schema_name: The schema the featurizer was trained for.
            iconq_query_featurizer_id: The identifier of the directory to load
            the IconqQueryFeaturizer from.
        """

        # Load parameters.
        load_dir = os.path.join(
            pu.get_data_path(),
            "__query_featurizations",
            schema_name,
            iconq_query_featurizer_id,
        )
        param_path = os.path.join(load_dir, "params.yml")
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        # Load featurization cache.
        cache_path = os.path.join(load_dir, "featurizations.yml")
        with open(cache_path, "r") as f:
            cache_list = yaml.safe_load(f)
        precomputed_featurization_cache = {
            QueryTextId(item["query_text_id"]): item["featurization"]
            for item in cache_list
        }

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

        return featurizer

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
            top_table_names = [
                table_name for (table_name, _) in self._top_tables
            ]
            if (base_table_name is not None) and (
                base_table_name in top_table_names
            ):
                table_idx = top_table_names.index(base_table_name)
                feature_idx = 2 * self._m + table_idx
                features[feature_idx] += float(cardinality)

        # Transform cardinality features, if needed.
        features = self._transform_card_if_needed(features)
        return features
