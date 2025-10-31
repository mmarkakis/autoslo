from typing import Optional

from psycopg2.extensions import connection as _Conn


def form_hostname(workgroup_name: str, aws_account_id: str, aws_region: str) -> str:
    """
    Form the hostname for connecting to a Redshift Serverless endpoint.

    Parameters:
        workgroup_name: The name of the Redshift Serverless workgroup.
        aws_account_id: The AWS account ID.
        aws_region: The AWS region.
    """
    return f"{workgroup_name}.{aws_account_id}.{aws_region}.redshift-serverless.amazonaws.com"


class ConnWithSetup(_Conn):
    """
    A subclass of psycopg2 connection that sets some session parameters on connect.
    """

    ONE_HOUR_MS = 60 * 60 * 1000  # One hour in milliseconds

    def __init__(
        self,
        *args,
        autocommit: bool = True,
        search_path: Optional[str] = None,
        statement_timeout: int = ONE_HOUR_MS,
        **kwargs,
    ):
        """
        Initialize the connection with custom session parameters.

        Parameters:
            autocommit: Whether to set autocommit mode.
            search_path: Schema search path to set.
            statement_timeout: Statement timeout in milliseconds.
        """

        super().__init__(*args, **kwargs)
        self.autocommit = autocommit
        with self.cursor() as cur:
            cur.execute("SET enable_result_cache_for_session TO OFF;")
            cur.execute(f"SET statement_timeout TO {statement_timeout};")
            if search_path:
                cur.execute(f"SET search_path TO {search_path};")
