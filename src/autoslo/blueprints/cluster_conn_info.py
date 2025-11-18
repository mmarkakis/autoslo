from dataclasses import dataclass


@dataclass
class ClusterConnInfo:
    """Dataclass to hold cluster connection information."""

    host: str
    port: int
    dbname: str
    user: str
    password: str

    @staticmethod
    def form_hostname(
        workgroup_name: str, aws_account_id: str, aws_region: str
    ) -> str:
        """
        Form the hostname for connecting to a Redshift Serverless endpoint.

        Parameters:
            workgroup_name: The name of the Redshift Serverless workgroup.
            aws_account_id: The AWS account ID.
            aws_region: The AWS region.
        """
        return ".".join(
            [
                workgroup_name,
                aws_account_id,
                aws_region,
                "redshift-serverless",
                "amazonaws",
                "com",
            ]
        )

    @staticmethod
    def from_dict(d: dict) -> "ClusterConnInfo":
        """
        Create a ClusterConnInfo instance from a dictionary.

        Parameters:
            d: A dictionary containing the cluster connection information.
                Expected keys are 'host', 'port', 'dbname', 'user' and
                'password'. If host is missing, it will be derived from
                the 'workgroup_name', 'aws_account_id', and 'aws_region' keys,
                if all are present.

        Returns:
            A ClusterConnInfo instance created from the dictionary.

        Raises:
            KeyError: If required keys are missing from the dictionary.
        """

        if "host" not in d:
            if not all(
                key in d
                for key in ("workgroup_name", "aws_account_id", "aws_region")
            ):
                raise KeyError(
                    "Missing 'host' and insufficient information to derive it."
                )
            host = ClusterConnInfo.form_hostname(
                workgroup_name=d["workgroup_name"],
                aws_account_id=str(d["aws_account_id"]),
                aws_region=d["aws_region"],
            )
        else:
            host = d["host"]

        for key in ("port", "dbname", "user", "password"):
            if key not in d:
                raise KeyError(
                    f"Missing required key '{key}' in connection info."
                )

        return ClusterConnInfo(
            host=host,
            port=d["port"],
            dbname=d["dbname"],
            user=d["user"],
            password=d["password"],
        )
