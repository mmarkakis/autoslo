"""
The code in this file is derived from https://github.com/DataManagementLab/zero-shot-cost-estimation
"""

import math
import re
import logging
from typing import Optional, Any, cast
import numpy as np
from decimal import Decimal

logger = logging.getLogger(__name__)

# Regexes for parsing the query plan
estimated_regex = re.compile(
    r"""\(cost=(?P<est_startup_cost>\d+.\d+)\.\.(?P<est_cost>\d+.\d+)[ ]
    rows=(?P<est_card>\d+)[ ]width=(?P<est_width>\d+)\)""",
    re.VERBOSE,  # N.B. This allows us to split the regex into multiple lines
)
actual_regex = re.compile(
    r"\(actual time=(?P<act_startup_cost>\d+.\d+)\.\.(?P<act_time>\d+.\d+) rows=(?P<act_card>\d+)"
)
op_name_regex = re.compile(r'->  ([^"(]+)')
workers_planned_regex = re.compile(r"Workers Planned: (\d+)")
filter_columns_regex = re.compile(r"([^\(\)\*\+\-'\= ]+)")
literal_regex = re.compile(r"('[^']+'::[^'\)]+)")


class PlanOperator(dict):
    """
    A class to represent a single operator in a query plan.
    """

    def __init__(
        self,
        plain_content: list[str],
        children: Optional[list["PlanOperator"]] = None,
        plan_parameters: Optional[dict[str, Any]] = None,
        plan_runtime: float = 0,
    ):
        super().__init__()
        self.__dict__ = self
        self.plain_content = plain_content

        self.plan_parameters = (
            plan_parameters if (plan_parameters is not None) else {}
        )
        self.children = list(children) if (children is not None) else []
        self.plan_runtime = plan_runtime

    def parse_lines(
        self,
        schema_name: Optional[str] = None,
        alias_dict: Optional[dict[str, Optional[str]]] = None,
    ) -> None:
        """
        Parse the lines of the operator.

        Parameters:
            schema_name: The schema name to use for table names.
            alias_dict: A dictionary mapping table aliases to their actual table names.
        """

        op_line = self.plain_content[0]

        # Parse plan operator name
        op_name_match = op_name_regex.search(op_line)
        assert op_name_match is not None
        op_name = op_name_match.groups()[0]
        for split_word in ["on", "using"]:
            if f" {split_word} " in op_name:
                op_name = op_name.split(f" {split_word} ")[0]
        op_name = op_name.strip()

        # Operator table
        if " on " in op_line or (
            (schema_name is not None) and (f"{schema_name}." in op_name)
        ):
            table_name = ""
            if " on " in op_line:
                table_name = op_line.split(" on ")[1].strip()
            else:
                table_name = op_name.split(f"{schema_name}.")[1].strip()
            table_name_parts = table_name.split(" ")

            table_name = table_name_parts[0].strip('"')

            if table_name.endswith("_pkey"):
                table_name = table_name.replace("_pkey", "")

            if "." in table_name:
                table_name = table_name.split(".")[1].strip('"')

            if len(table_name_parts) > 1 and alias_dict is not None:
                potential_alias = table_name_parts[1]
                if potential_alias != "" and not potential_alias.startswith(
                    "("
                ):
                    alias_dict[potential_alias] = table_name
                    self.plan_parameters.update({"alias": potential_alias})

            if "Subquery Scan" in op_line and alias_dict is not None:
                alias_dict[table_name] = None
            else:
                self.plan_parameters.update({"table": table_name})

        self.plan_parameters.update({"op_name": op_name})

        # Parse estimated plan costs
        match_est = estimated_regex.search(op_line)
        assert match_est is not None
        self.plan_parameters.update(
            {k: float(v) for k, v in match_est.groupdict().items()}
        )

        # Parse actual plan costs
        match_act = actual_regex.search(op_line)
        if match_act is not None:
            self.plan_parameters.update(
                {k: float(v) for k, v in match_act.groupdict().items()}
            )

        # Collect additional optional information
        for l in self.plain_content[1:]:
            l = l.strip()
            workers_planned_match = workers_planned_regex.search(l)

            if workers_planned_match is None:
                continue

            workers_planned = workers_planned_match.groups()
            if isinstance(workers_planned, (list, tuple)):
                workers_planned = workers_planned[0]
            num_workers_planned = int(cast(Any, workers_planned))
            self.plan_parameters.update(
                {"workers_planned": num_workers_planned}
            )
        self.plain_content = []

    def parse_columns_bottom_up(
        self,
        alias_dict: Optional[dict[str, Optional[str]]],
    ) -> set[str]:
        """
        Parse the columns of the operator and its children bottom-up.

        Parameters:
            alias_dict: A dictionary mapping table aliases to their actual table names.

        Returns:
            The set of tables considered at this node.

        Raises:
            ValueError: If a column cannot be uniquely identified.
        """

        if alias_dict is None:
            alias_dict = {}

        # First keep track which tables are actually considered here
        node_tables: set[str] = set()
        table = cast(str, self.plan_parameters.get("table"))
        if (table is not None) and (not table.startswith("volt_tt_")):
            node_tables.add(table)
        for c in self.children:
            node_tables.update(
                c.parse_columns_bottom_up(
                    alias_dict,
                )
            )

        # Process child cardinalities
        self.plan_parameters["act_children_card"] = self.child_prod("act_card")
        self.plan_parameters["est_children_card"] = self.child_prod("est_card")

        return node_tables

    def child_prod(self, feature_name: str, default: float = 1.0) -> float:
        """
        Compute the product of a feature over all children of the operator.

        Parameters:
            feature_name: The name of the feature to compute the product of.
            default: The default value to return if no children have the feature.

        Returns:
            The product of the feature over all children of the operator.
        """

        child_feats = [
            cast(float, c.plan_parameters.get(feature_name))
            for c in self.children
            if c.plan_parameters.get(feature_name) is not None
        ]
        if len(child_feats) == 0:
            return default
        return float(np.prod(child_feats))

    def merge_recursively(self, node: "PlanOperator") -> None:
        """
        Merge the operator with another operator recursively, by combining
        their plan parameters at each node.

        Parameters:
            node: The operator to merge with.
        """
        assert (
            self.plan_parameters["op_name"] == node.plan_parameters["op_name"]
        )
        assert len(self.children) == len(node.children)

        self.plan_parameters.update(node.plan_parameters)
        for self_c, c in zip(self.children, node.children):
            self_c.merge_recursively(c)

    def parse_lines_recursively(
        self,
        schema_name: Optional[str] = None,
        alias_dict: Optional[dict[str, Optional[str]]] = None,
    ):
        """
        Parse the lines of the operator and its children recursively.

        Parameters:
            schema_name: The schema name to use for table names.
            alias_dict: A dictionary mapping table aliases to their actual table names.
        """

        self.parse_lines(
            schema_name=schema_name,
            alias_dict=alias_dict,
        )
        for c in self.children:
            c.parse_lines_recursively(
                schema_name=schema_name, alias_dict=alias_dict
            )

    def min_card(self) -> float:
        """
        Recursively find the minimum cardinality ("act_card") of the operator and its children.

        Returns:
            The minimum cardinality of the operator and its children.
        """
        act_card = self.plan_parameters.get("act_card")
        if act_card is None:
            act_card = math.inf

        for c in self.children:
            child_min_card = c.min_card()
            if child_min_card < act_card:
                act_card = child_min_card

        if act_card == math.inf:
            act_card = -1

        return act_card

    def recursive_str(self, num_indentation_tabs: int) -> list[str]:
        """
        Recursively generate a string representation of the operator.

        Parameters:
            num_indentation_tabs: The number of tabs to indent the current level of the
                string representation by.

        Returns:
            A list of strings representing the operator and its children.
        """

        current_string = ("\t" * num_indentation_tabs) + str(
            self.plan_parameters
        )
        node_strings = [current_string]

        for c in self.children:
            node_strings.extend(c.recursive_str(num_indentation_tabs + 1))

        return node_strings

    def __str__(self) -> str:
        """
        A string representation of the operator.
        """
        rec_str = self.recursive_str(0)
        return "\n".join(rec_str)

    def as_serializable(self) -> dict[str, Any]:
        """
        Convert the operator to a serializable dictionary by converting anything
        of type set to a list, at every level.

        Returns:
            A serializable dictionary representation of the operator.
        """
        def _make_serializable(obj: Any) -> Any:
            # PlanOperator → dict
            if isinstance(obj, PlanOperator):
                return obj.as_serializable()

            # Decimal → float
            if isinstance(obj, Decimal):
                return float(obj)

            # set → list
            if isinstance(obj, set):
                return [_make_serializable(x) for x in obj]

            # list/tuple → list (YAML handles lists fine)
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(x) for x in obj]

            # dict → dict with string keys and serialized values
            if isinstance(obj, dict):
                return {
                    str(k): _make_serializable(v)
                    for k, v in obj.items()
                }

            # everything else left as-is
            return obj

        return {
            k: _make_serializable(v)
            for k, v in self.__dict__.items()
        }
