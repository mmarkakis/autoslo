import os
import yaml
import pytest

from chunkload.building_blocks.chunk import Chunk
import chunkload.utils.paths as pu


def test_constructor_validation_invalid_args():
    """
    Ensure that invalid constructor arguments raise ValueError.
    """
    # Invalid H
    with pytest.raises(ValueError):
        Chunk(H=-1, T=10)
    with pytest.raises(ValueError):
        Chunk(H=101, T=10)
    # Invalid T
    with pytest.raises(ValueError):
        Chunk(H=10, T=0)
    # Invalid schema
    with pytest.raises(ValueError):
        supported_schemas = Chunk.SUPPORTED_SCHEMAS
        # Generate an unsupported schema name based on supported ones
        unsupported_schema = "not_" + supported_schemas[0]
        Chunk(H=10, T=10, schema=unsupported_schema)
    # Invalid chunk duration
    with pytest.raises(ValueError):
        Chunk(H=10, T=10, chunk_duration_s=0)
    # Invalid templates / queries per template
    with pytest.raises(ValueError):
        Chunk(H=10, T=10, num_templates=0)
    with pytest.raises(ValueError):
        Chunk(H=10, T=10, num_queries_per_template=0)
    # Invalid stddev_interarrival_s
    with pytest.raises(ValueError):
        Chunk(H=10, T=10, stddev_interarrival_s=0)


def test_to_from_dict_roundtrip():
    """
    Check that to_dict produces a dict suitable for from_dict and that
    from_dict recreates equivalent Chunk parameters.
    """
    c1 = Chunk(H=25, T=60, schema="tpcds", chunk_duration_s=1800,
               random_seed=7, num_templates=5, num_queries_per_template=2,
               stddev_interarrival_s=30)
    d = c1.to_dict()
    c2 = Chunk.from_dict(d)
    assert c2.H == c1.H
    assert c2.T == c1.T
    assert c2.schema == c1.schema
    assert c2.chunk_duration_s == c1.chunk_duration_s
    assert c2.random_seed == c1.random_seed
    assert c2.num_templates == c1.num_templates
    assert c2.num_queries_per_template == c1.num_queries_per_template
    assert c2.stddev_interarrival_s == c1.stddev_interarrival_s


def test_save_writes_yaml(tmp_path, monkeypatch):
    """
    Verify that save() writes a chunk_definition.yaml in the expected
    save directory and that the YAML content matches to_dict().
    """
    # Redirect DATA_PATH to a temporary directory
    monkeypatch.setattr(pu, "DATA_PATH", str(tmp_path))
    c = Chunk(H=10, T=30, num_templates=3)
    c.save()
    out_dir = c.save_dir()
    yaml_path = os.path.join(out_dir, "chunk_definition.yaml")
    assert os.path.exists(yaml_path)
    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    # YAML loader may convert numeric types; compare dictionaries
    assert loaded == c.to_dict()


def test_color_and_shape_mappings():
    """
    Ensure that color() and shape() return values from the class mapping
    dictionaries for representative H and T values.
    """
    # color mapping: test T exactly at a threshold and below thresholds
    c_highT = Chunk(H=0, T=120)
    assert c_highT.color() == Chunk.T_COLOR_MAP[120]
    c_midT = Chunk(H=0, T=60)
    assert c_midT.color() == Chunk.T_COLOR_MAP[60]
    c_lowT = Chunk(H=0, T=10)
    assert c_lowT.color() == Chunk.T_COLOR_MAP[10]
    # shape mapping: test H exactly at thresholds and below
    c_h0 = Chunk(H=0, T=30)
    assert c_h0.shape() == Chunk.H_SHAPE_MAP[0]
    c_h25 = Chunk(H=25, T=30)
    assert c_h25.shape() == Chunk.H_SHAPE_MAP[25]
    c_h50 = Chunk(H=50, T=30)
    assert c_h50.shape() == Chunk.H_SHAPE_MAP[50]


def test_chunk_id_string_format():
    """
    Confirm that the chunk_id attribute contains the expected formatted
    components (schema, templates, H and T formatted parts).
    """
    c = Chunk(H=5, T=7, schema="tpcds", num_templates=2)
    # chunk.chunk_id is stored as an attribute string
    chunk_id = c.chunk_id()
    assert isinstance(chunk_id, str)
    assert "tpcds" in chunk_id
    assert "2templates" in chunk_id
    assert "05pctheavy" in chunk_id
    # T is formatted with two digits in the chunk_id
    assert "07meaninterarrivals" in chunk_id
    