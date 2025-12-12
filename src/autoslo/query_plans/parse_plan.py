"""
The code in this file is derived from https://github.com/DataManagementLab/zero-shot-cost-estimation
"""

import re
from typing import Any, Optional, Sequence, cast

from autoslo.query_plans.plan_operator import PlanOperator

# Plan parsing regexes
planning_time_regex = re.compile(
    r"planning time: (?P<planning_time>\d+.\d+) ms"
)
ex_time_regex = re.compile(r"execution time: (?P<execution_time>\d+.\d+) ms")
init_plan_regex = re.compile(r"InitPlan \d+ \(returns \$\d\)")
join_columns_regex = re.compile(r"\w+\.\w+ ?= ?\w+\.\w+")


def list_columns(n, s: set[tuple[str, str]]) -> None:
    """
    Recursively list the column names and operators in the filter conditions of the plan.

    Parameters:
        n: The current node in the plan.
        s: The set to add the column names and operators to.
    """
    s.add(
        (n.column, n.operator)
    )  # N.B.: n is of type PredicateNode form the original repo.
    for c in n.children:
        list_columns(c, s)


def plan_summary(
    plan_op: PlanOperator,
    tables: Optional[set[str]] = None,
    filter_columns: Optional[set[tuple[str, str]]] = None,
    operators: Optional[set[str]] = None,
    skip_columns=False,
    conv_to_dict=False,
) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    """
    Get a summary of the plan, including the tables, filter columns, and operators.

    Parameters:
        plan_op: The root operator of the plan.
        tables: The set of tables in the plan.
        filter_columns: The set of filter columns in the plan.
        operators: The set of operators in the plan.
        skip_columns: Whether to skip listing the columns.
        conv_to_dict: Whether to convert the parameters to a dictionary.

    Returns:
        tables: The set of tables in the plan.
        filter_columns: The set of filter columns in the plan.
        operators: The set of operators in the plan.
    """

    # Set default values
    if tables is None:
        tables = set()
    if operators is None:
        operators = set()
    if filter_columns is None:
        filter_columns = set()

    # Get the parameters of the current operator
    params = plan_op.plan_parameters
    if conv_to_dict:
        params = vars(params)
    if "table" in params:
        tables.add(params["table"])
    if "op_name" in params:
        operators.add(params["op_name"])
    if "filter_columns" in params and not skip_columns:
        list_columns(params["filter_columns"], filter_columns)

    # Recursively get the parameters of the children
    for c in plan_op.children:
        plan_summary(
            c,
            tables=tables,
            filter_columns=filter_columns,
            operators=operators,
            skip_columns=skip_columns,
            conv_to_dict=conv_to_dict,
        )

    return tables, filter_columns, operators


def maybe_create_node(
    lines_plan_operator: list[str], operators_current_level: list[PlanOperator]
) -> None:
    """
    Create a PlanOperator object from the lines of the plan and add it to the list of operators.

    Parameters:
        lines_plan_operator: The lines of the plan that describe the operator.
        operators_current_level: The list of operators at the current level of the plan.
    """
    if len(lines_plan_operator) > 0:
        last_operator = PlanOperator(lines_plan_operator)
        operators_current_level.append(last_operator)


def count_left_whitespaces(a: str) -> int:
    """
    Count the number of whitespaces at the beginning of a string.

    Parameters:
        a: The string to count the whitespaces in.

    Returns:
        The number of whitespaces at the beginning of the string.
    """
    idx = 0
    while idx < len(a) and a[idx] == " ":
        idx += 1
    return idx


def parse_recursively(
    parent: Optional[PlanOperator], plan: list[str], offset: int, depth: int
) -> tuple[Optional[int], Optional[PlanOperator]]:
    """
    Recursively parse the plan into a tree of PlanOperator objects.

    Parameters:
        parent: The parent of the current operator.
        plan: The lines of the plan.
        offset: The current offset in the plan (next line to parse).
        depth: The current depth in the plan (indentation level).

    Returns:
        next_offset: The next offset in the plan (next line to parse). Only
            returned if there are more lines to parse.
        root_operator: The root operator of the tree. Only returned if the
            current call is at the top level of the plan.
    """
    lines_plan_operator: list[str] = []
    i = offset
    operators_current_level: list[PlanOperator] = []

    while i < len(plan):
        # new operator
        if plan[i].strip().startswith("->"):
            # create plan node for previous one
            maybe_create_node(lines_plan_operator, operators_current_level)
            lines_plan_operator = []

            new_depth = count_left_whitespaces(plan[i])

            # One step down in recursion
            if new_depth > depth:
                assert (
                    len(operators_current_level) > 0
                ), "No parent found at this level"
                j, _ = parse_recursively(
                    operators_current_level[-1], plan, i, new_depth
                )
                i = cast(int, j)  # For type checker.

            # One step up in recursion
            elif new_depth < depth:
                break

            # new operator in current depth
            elif new_depth == depth:
                lines_plan_operator.append(plan[i])
                i += 1

        else:
            lines_plan_operator.append(plan[i])
            i += 1

    # Create plan node for the last operator
    maybe_create_node(lines_plan_operator, operators_current_level)

    # Set parent's children if needed.
    if parent is not None:
        parent.children = operators_current_level
        return i, None

    # There should only be one top node.
    assert len(operators_current_level) == 1
    return None, operators_current_level[0]


def parse_one_plan(
    plan_steps: list[str], analyze: bool = True
) -> tuple[PlanOperator, float, float]:
    """
    Parse one plan from the text format the database returns it in, into
    a tree of PlanOperator objects.

    Parameters:
        plan_steps: The lines of the plan.
        analyze: Whether the plan is from an EXPLAIN ANALYZE command.

    Returns:
        root_operator: The root operator of the tree.
        ex_time: The execution time of the query.
        planning_time: The planning time of the query.
    """

    # Transform input.
    plan_steps[0] = (
        f"->  {plan_steps[0]}"  # For some reason this is missing in postgres.
    )

    # If this comes from an EXPLAIN ANALYZE, extract the planing and execution times.
    # These should be reported in the last two lines of the plan, in this order.
    ex_time = 0.0
    planning_time = 0.0
    planning_idx = -1
    if analyze:
        for i, plan_step in enumerate(plan_steps):
            plan_step = plan_step.lower()

            planning_time_match = planning_time_regex.match(plan_step)
            if planning_time_match is not None:
                planning_idx = i
                planning_time = float(planning_time_match.groups()[0])

            ex_time_match = ex_time_regex.match(plan_step)
            if ex_time_match is not None:
                ex_time = float(ex_time_match.groups()[0])

        assert ex_time > 0 and planning_time > 0
        plan_steps = plan_steps[:planning_idx]

    # Parse the plan
    _, root_operator = parse_recursively(None, plan_steps, 0, 0)

    return cast(PlanOperator, root_operator), ex_time, planning_time
