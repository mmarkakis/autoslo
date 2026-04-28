from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class PartialConfig:
    """
    A parent class for partial configs that implements parsing from a config
    file.
    """

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> Self:
        """
        Parse the relevant fields from the given config dict and return an
        instance of the PartialConfig subclass.
        """

        # Go from camel case to snake case to find the relevant section.
        sub_config_name = cls.__name__  # e.g. "AutoscalerConfig"
        sub_config_key = "".join(
            ["_" + c.lower() if c.isupper() else c for c in sub_config_name]
        ).lstrip(
            "_"
        )  # e.g. "autoscaler_config"

        if sub_config_key in cfg:
            cfg = cfg[sub_config_key]
        return cls(**cfg, **kwargs)
