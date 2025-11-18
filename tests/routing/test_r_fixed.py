from typing import List

import pytest

from autoslo.routing.r_fixed import RFixed



class DummyBlueprint:
	# minimal blueprint replacement exposing cluster_names
	def __init__(self, cluster_names: List[str]) -> None:
		self.cluster_names = cluster_names


def test_rfixed_routes_to_fixed_cluster() -> None:
	"""
	Verify route_query returns the fixed cluster name for any query.
	"""
	blueprint = DummyBlueprint(["clusterA", "clusterB"]) 
	rf = RFixed(blueprint, "clusterA") # type: ignore
	result = rf.route_query("SELECT 1")
	assert result == "clusterA"


def test_rfixed_raises_for_missing_cluster() -> None:
	"""
	Verify constructing RFixed with a non-existent cluster raises.
	"""
	blueprint = DummyBlueprint(["clusterX"])
	with pytest.raises(ValueError):
		RFixed(blueprint, "clusterA")  # type: ignore


def test_name_contains_fixed_cluster_repr() -> None:
	"""
	Verify the name property includes the repr of the cluster name.
	"""
	blueprint = DummyBlueprint(["clusterA"])
	rf = RFixed(blueprint, "clusterA")  # type: ignore
	expected = f"RFixed(fixed_cluster_name={repr('clusterA')})"
	assert rf.name == expected
