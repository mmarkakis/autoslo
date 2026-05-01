from enum import Enum


class QueryRouterPolicy(Enum):
    USE_STAGE_MODEL = "use_stage_model"
    USE_ICONQ_MODEL = "use_iconq_model"
    ROUND_ROBIN = "round_robin"
    UNIFORM_RANDOM = "uniform_random"
    CACHE_AWARE = "cache_aware"
