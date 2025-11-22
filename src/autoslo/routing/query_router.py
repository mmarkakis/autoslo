import ast
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from autoslo.blueprints.blueprint import Blueprint


class QueryRouter(ABC):
    """
    An abstract base class for routing queries to clusters within the context
    of a blueprint. Can be instantiated via the from_name factory method.
    """

    # registry mapping class name -> subclass
    _registry: Dict[str, type["QueryRouter"]] = {}

    def __init_subclass__(cls, **kwargs):
        # auto-register subclasses by their class name
        super().__init_subclass__(**kwargs)
        QueryRouter._registry[cls.__name__] = cls

    # manual registration decorator / function
    @classmethod
    def register(
        cls, subcls: Optional[type] = None, *, name: Optional[str] = None
    ):
        """
        Register a subclass manually.

        Parameters:
            subcls: The subclass to register. If None, returns a decorator.
            name: Optional name to register the subclass under. If None, uses
                the subclass's class name.

        Returns:
            If subcls is None, returns a decorator that registers the subclass.
            Otherwise, returns the registered subclass.

        Raises:
            TypeError: If subcls is not a subclass of QueryRouter.
        """
        if subcls is None:
            return lambda sc: cls.register(sc, name=name)
        key = name or subcls.__name__
        assert subcls is not None

        # Assert that subcls is indeed a subclass of QueryRouter
        if not issubclass(subcls, QueryRouter):
            raise TypeError(
                f"Cannot register {subcls.__name__}: not a subclass "
                "of QueryRouter"
            )
        else:
            cls._registry[key] = subcls
        return subcls

    @classmethod
    def ensure_registry_populated(
        cls, package: Optional[str] = "autoslo.routing"
    ) -> None:
        """
        Import modules under `package` to trigger subclass registration.
        - If package is None, does nothing.
        - Ignores import errors in submodules.
        Use this before calling from_name when you can't guarantee modules were
        imported.

        Parameters:
            package: The package under which to look for routing submodules.
        """
        if not package:
            return
        try:
            pkg = importlib.import_module(package)
        except Exception:
            # package not importable; nothing to import
            return
        path = getattr(pkg, "__path__", None)
        if not path:
            return
        for finder, modname, ispkg in pkgutil.iter_modules(path):
            fullname = f"{package}.{modname}"
            try:
                importlib.import_module(fullname)
            except Exception:
                # ignore errors - modules may be optional or have extra deps
                continue

    @classmethod
    def registered_names(cls) -> List[str]:
        """Return a list of currently-registered router keys."""
        return list(cls._registry.keys())
    
    @classmethod
    def from_name(
        cls,
        name: str,
        blueprint: Blueprint,
        *args,
        auto_populate: bool = True,
        package: Optional[str] = "autoslo.routing",
        **kwargs,
    ) -> "QueryRouter":
        """
        Factory: instantiate a registered QueryRouter subclass from its name.

        Supported name formats:
          - "ClassName"           -> calls ClassName(blueprint, *args, **kwargs)
          - "ClassName(pos1, pos2)"    -> positional args passed after blueprint
          - "ClassName(key=val, key2=val2)"    -> keyword args parsed and passed
          - mixed positional and keyword allowed: "ClassName(pos, key=val)"

        Simple literal parsing is attempted via ast.literal_eval; if it fails,
        the value is used as an unquoted string.

        Parameters:
            name: The string name representing the QueryRouter subclass and its
                initialization parameters.
            blueprint: The Blueprint instance to be passed as the first argument
                to the subclass constructor.
            *args: Additional positional arguments, as needed.
            auto_populate: If True, automatically populate the registry by
                importing routing submodules before lookup.
            package: The package under which to look for routing submodules.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            An instance of the specified QueryRouter subclass.

        Raises:
            ValueError: If the class name is not registered or if the name
                string is malformed.
            ValueError: If there is an error instantiating the subclass with the
                provided arguments.
        """
        # optionally populate registry by importing routing submodules
        if auto_populate:
            cls.ensure_registry_populated(package)

        if "(" not in name:
            class_name = name
            inside = ""
        else:
            open_idx = name.find("(")
            close_idx = name.rfind(")")
            class_name = name[:open_idx].strip()
            inside = name[open_idx + 1 : close_idx].strip()

        if class_name not in cls._registry:
            raise ValueError(f"Unknown QueryRouter subclass '{class_name}'")

        positional: List[Any] = []
        kw: Dict[str, Any] = {}

        if inside:
            # split on commas (simple splitter; expects no nested commas)
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        parsed = ast.literal_eval(v)
                    except Exception:
                        # fall back to raw string (strip surrounding quotes)
                        parsed = v.strip("'\"")
                    kw[k] = parsed
                else:
                    # positional arg
                    v = part
                    try:
                        parsed = ast.literal_eval(v)
                    except Exception:
                        parsed = v.strip("'\"")
                    positional.append(parsed)

        subcls = cls._registry[class_name]
        # instantiate subclass with blueprint and parsed args.
        try:
            subclass_instance = subcls(blueprint, *positional, **kw)
        except Exception as e:
            raise ValueError(
                f"Error instantiating QueryRouter subclass '{class_name}' "
                f"with args {positional} and kwargs {kw}: {e}"
            ) from e
        return subclass_instance

    def __init__(self, blueprint: Blueprint, *args, **kwargs) -> None:
        """
        Initialize a QueryRouter instance.

        Parameters:
            blueprint: The Blueprint instance containing the clusters to route
                queries to.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        self._blueprint = blueprint

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this QueryRouter.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the QueryRouter instance. Should reflect both the type
        of router and any key configuration parameters.

        Returns:
            The name of the QueryRouter instance.
        """
        pass

    @abstractmethod
    def route_query(self, query: str, *args, **kwargs) -> str:
        """
        Given a query string, determine the appropriate cluster to route it to.

        Parameters:
            query: The SQL query string to be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """
        pass
