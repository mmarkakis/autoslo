from typing import Any

import pytest

from autoslo.routing.query_router import QueryRouter


class DummyBlueprint:
    # minimal blueprint replacement exposing cluster_names
    def __init__(self, cluster_names: list[str]) -> None:
        self.cluster_names = cluster_names


def test_from_name_instantiates_simple_name() -> None:
    """
    Verify that QueryRouter.from_name instantiates a simple subclass by
    name when auto_populate is disabled.
    """

    class SimpleRouter(QueryRouter):
        def __init__(self, blueprint: DummyBlueprint) -> None:
            super().__init__(blueprint)  # type: ignore
            self.flag = True

        def route_query(self, query: str, *args: Any, **kwargs: Any) -> str:
            return "unused"

        @property
        def name(self) -> str:
            return "SimpleRouter()"

    blueprint = DummyBlueprint(["c"])
    obj = QueryRouter.from_name(
        "SimpleRouter", blueprint, auto_populate=False  # type: ignore
    )
    assert isinstance(obj, SimpleRouter)
    assert obj.blueprint is blueprint


def test_from_name_parses_args_and_kwargs() -> None:
    """
    Verify that positional and keyword arguments are parsed from the name
    string and passed to the subclass constructor.
    """

    class ParamRouter(QueryRouter):
        def __init__(
            self, blueprint: DummyBlueprint, x: str, y: int = 0
        ) -> None:
            super().__init__(blueprint)  # type: ignore
            self.x = x
            self.y = y

        def route_query(self, query: str, *args: Any, **kwargs: Any) -> str:
            return "unused"

        @property
        def name(self) -> str:
            return f"ParamRouter(x={repr(self.x)}, y={self.y})"

    blueprint = DummyBlueprint(["c"])
    obj = QueryRouter.from_name(
        "ParamRouter('hi', y=5)", blueprint, auto_populate=False  # type: ignore
    )
    assert isinstance(obj, ParamRouter)
    assert obj.x == "hi"
    assert obj.y == 5


def test_from_name_unknown_class_raises() -> None:
    """
    Verify that requesting an unknown router name raises ValueError.
    """
    blueprint = DummyBlueprint(["c"])
    with pytest.raises(ValueError):
        QueryRouter.from_name(
            "NoSuchClass", blueprint, auto_populate=False  # type: ignore
        )


def test_register_decorator_allows_custom_name() -> None:
    """
    Verify that QueryRouter.register can register a subclass under a
    custom name used by from_name.
    """

    @QueryRouter.register(name="CustomRoute")
    class DecoratedRouter(QueryRouter):
        def __init__(self, blueprint: DummyBlueprint) -> None:
            super().__init__(blueprint)  # type: ignore

        def route_query(self, query: str, *args: Any, **kwargs: Any) -> str:
            return "unused"

        @property
        def name(self) -> str:
            return "CustomRoute()"

    blueprint = DummyBlueprint(["c"])
    obj = QueryRouter.from_name(
        "CustomRoute", blueprint, auto_populate=False  # type: ignore
    )
    assert isinstance(obj, DecoratedRouter)
