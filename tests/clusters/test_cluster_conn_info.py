import pytest

from autoslo.clusters.cluster_conn_info import ClusterConnInfo


def test_form_hostname():
    workgroup = "wg"
    account = "123456789012"
    region = "us-east-1"
    expected = f"{workgroup}.{account}.{region}.redshift-serverless.amazonaws.com"
    assert ClusterConnInfo.form_hostname(workgroup, account, region) == expected


def test_from_dict_derives_host_and_fields():
    d = {
        "workgroup_name": "wg",
        "aws_account_id": 123456789012,
        "aws_region": "us-east-1",
        "port": 5439,
        "dbname": "db",
        "user": "u",
        "password": "p",
    }
    ci = ClusterConnInfo.from_dict(d)
    assert ci.host == ClusterConnInfo.form_hostname(
        workgroup_name="wg",
        aws_account_id=str(123456789012),
        aws_region="us-east-1",
    )
    assert ci.port == 5439
    assert ci.dbname == "db"
    assert ci.user == "u"
    assert ci.password == "p"


def test_from_dict_uses_explicit_host_when_present():
    d = {
        "host": "explicit-host.example.com",
        "port": 5439,
        "dbname": "db",
        "user": "u",
        "password": "p",
    }
    ci = ClusterConnInfo.from_dict(d)
    assert ci.host == "explicit-host.example.com"
    assert ci.port == 5439


def test_from_dict_missing_host_and_workgroup_raises():
    d = {
        # missing 'host' and missing required workgroup/aws keys
        "port": 5439,
        "dbname": "db",
        "user": "u",
        "password": "p",
    }
    with pytest.raises(KeyError):
        ClusterConnInfo.from_dict(d)
