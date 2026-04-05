"""
Base class with registry and factory methods for AutoSLO components.
"""

import ast
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, Self


class ClassWithFactory(ABC):

    # Each sublcass of ClassWithFactory should have its own _registry variable,
    # where it maps strings to the types of *its* subclasses.
    _registry: dict[str, type[Self]] = {}

    # Add an init_subclass to populate the registry automatically when a
    # subclass is defined.
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_registry"):
            cls._registry = {}
        # Register the subclass using its class name.
        cls._registry[cls.__name__] = cls

    @classmethod
    def ensure_registry_populated(
        cls,
    ) -> None:
        """
        Import modules under `package` to trigger subclass registration.
        - If package is None, does nothing.
        - Ignores import errors in submodules.
        Use this before calling from_name when you can't guarantee modules were
        imported.
        """
        package_name = cls.__module__.rsplit(".", 1)[0]

        try:
            pkg = importlib.import_module(package_name)
        except Exception:
            # package not importable; nothing to import
            return
        path = getattr(pkg, "__path__", None)
        if not path:
            return
        for finder, modname, ispkg in pkgutil.iter_modules(path):
            fullname = f"{package_name}.{modname}"
            try:
                importlib.import_module(fullname)
            except Exception:
                # ignore errors - modules may be optional or have extra deps
                continue

    @classmethod
    def from_name(
        cls,
        name: str,
        *args,
        **kwargs,
    ) -> Self:
        """
        Factory: instantiate a registered subclass from its name.

        Supported name format:
            - "ClassName(key=val, key2=val2)"

        Simple literal parsing is attempted via ast.literal_eval; if it fails,
        the value is used as an unquoted string.

        Parameters:
            cls: The base class type.
            name: The string name representing the subclass and its
                initialization parameters.
            *args: Additional positional arguments for the subclass, as needed.
            **kwargs: Additional keyword arguments for the subclass, as needed.

        Returns:
            An instance of the specified subclass.

        Raises:
            ValueError: If the class name is not registered or if the name
                string is malformed.
            ValueError: If there is an error instantiating the subclass with the
                provided arguments.
        """
        # Ensure registry is populated
        cls.ensure_registry_populated()

        if "(" not in name:
            class_name = name
            inside = ""
        else:
            open_idx = name.find("(")
            close_idx = name.rfind(")")
            class_name = name[:open_idx].strip()
            inside = name[open_idx + 1 : close_idx].strip()

        if class_name not in cls._registry:
            raise ValueError(f"Unknown subclass '{class_name}'")

        positional: list[Any] = list(args)
        kw: dict[str, Any] = dict(kwargs)

        if inside:
            # split on commas (simple splitter; expects no nested commas)
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            for part in parts:
                if not "=" in part:
                    raise ValueError(f"Malformed name string: '{name}'")
                else:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        parsed = ast.literal_eval(v)
                    except Exception:
                        # fall back to raw string (strip surrounding quotes)
                        parsed = v.strip("'\"")
                    kw[k] = parsed

        subcls = cls._registry[class_name]
        # instantiate subclass with blueprint and parsed args.
        try:
            subclass_instance = subcls(*positional, **kw)
        except Exception as e:
            raise ValueError(
                f"Error instantiating {cls.__name__} subclass '{class_name}' "
                f"with args {positional} and kwargs {kw}: {e}"
            ) from e
        return subclass_instance

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the ClassWithFactory instance. Should reflect both the
        type of the bottom-level subclass and any key configuration parameters.

        Returns:
            The name of the ClassWithFactory instance.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'name' property."
        )


    def __str__(self) -> str:
        return self.name
