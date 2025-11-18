from typing import Optional

from psycopg2.extensions import connection as _Conn


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
