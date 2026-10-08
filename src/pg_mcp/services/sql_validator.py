"""SQL Security Validator using SQLGlot.

This module provides SQL validation and security checking using SQLGlot parser.
It ensures that only safe, read-only queries are executed and blocks potentially
dangerous operations.
"""

import re
from typing import ClassVar

import sqlglot
from sqlglot import exp

from pg_mcp.config.settings import SecurityConfig
from pg_mcp.models.errors import SecurityViolationError, SQLParseError


class SQLValidator:
    """SQL security validator using SQLGlot for parsing and validation.

    This validator ensures queries are safe by:
    - Allowing only SELECT statements
    - Blocking dangerous functions (pg_sleep, file operations, etc.)
    - Preventing access to blocked tables and columns
    - Rejecting multi-statement queries
    - Validating subquery safety
    """

    # Allowed statement types at the top level (including set operations)
    ALLOWED_STATEMENT_TYPES: ClassVar = {exp.Select, exp.Union, exp.Intersect, exp.Except}

    # Allowed top-level expressions (including CTEs)
    ALLOWED_TOP_LEVEL: ClassVar = {
        exp.Select,
        exp.Union,
        exp.Intersect,
        exp.Except,
        exp.With,
        exp.Subquery,
    }

    # Forbidden statement types
    FORBIDDEN_STATEMENT_TYPES: ClassVar = {
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Grant,
        exp.Revoke,
        exp.Set,
        exp.Command,
        exp.Use,
        exp.Merge,
    }

    # Built-in dangerous PostgreSQL functions
    BUILTIN_DANGEROUS_FUNCTIONS: ClassVar = {
        "pg_sleep",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "dblink_open",
        "pg_write_file",
        "pg_execute_sql",
        "copy_from",
        "copy_to",
    }

    def __init__(
        self,
        config: SecurityConfig,
        blocked_tables: list[str] | None = None,
        blocked_columns: list[str] | None = None,
        allow_explain: bool | None = None,
        blocked_schemas: list[str] | None = None,
        allowed_schemas: list[str] | None = None,
        allowed_tables: list[str] | None = None,
        allowed_columns: list[str] | None = None,
        allow_explain_analyze: bool | None = None,
    ) -> None:
        """Initialize SQL validator.

        Args:
            config: Security configuration containing blocked functions and settings.
            blocked_tables: Optional list of table names to block access to.
            blocked_columns: Optional list of column names to block access to.
            allow_explain: Whether to allow EXPLAIN statements.
        """
        self.config = config
        self.blocked_schemas = {
            s.lower()
            for s in (blocked_schemas if blocked_schemas is not None else config.blocked_schemas)
        }
        self.blocked_tables = {
            t.lower()
            for t in (blocked_tables if blocked_tables is not None else config.blocked_tables)
        }
        self.blocked_columns = {
            c.lower()
            for c in (blocked_columns if blocked_columns is not None else config.blocked_columns)
        }
        self.allowed_schemas = {
            s.lower()
            for s in (allowed_schemas if allowed_schemas is not None else config.allowed_schemas)
        }
        self.allowed_tables = {
            t.lower()
            for t in (allowed_tables if allowed_tables is not None else config.allowed_tables)
        }
        self.allowed_columns = {
            c.lower()
            for c in (allowed_columns if allowed_columns is not None else config.allowed_columns)
        }
        self.allow_explain = config.allow_explain if allow_explain is None else allow_explain
        self.allow_explain_analyze = (
            config.allow_explain_analyze if allow_explain_analyze is None else allow_explain_analyze
        )

        # Combine built-in dangerous functions with custom blocked functions
        self.blocked_functions = self.BUILTIN_DANGEROUS_FUNCTIONS | {
            f.lower() for f in config.blocked_functions
        }

    def validate(self, sql: str) -> tuple[bool, str | None]:
        """Validate SQL query for security compliance.

        Args:
            sql: SQL query string to validate.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        try:
            self.validate_or_raise(sql)
            return (True, None)
        except (SecurityViolationError, SQLParseError) as e:
            return (False, str(e))

    def validate_or_raise(self, sql: str) -> None:
        """Validate SQL query and raise exception on violation.

        Args:
            sql: SQL query string to validate.

        Raises:
            SQLParseError: If SQL cannot be parsed.
            SecurityViolationError: If SQL violates security constraints.
        """
        # Check for empty or whitespace-only SQL
        if not sql or not sql.strip():
            raise SQLParseError("SQL query cannot be empty")

        raw_sql = sql.strip().rstrip(";").strip()
        explain_match = re.match(
            r"^EXPLAIN(?:\s*\((?P<options>[^)]*)\))?\s+(?P<inner>.+)$",
            raw_sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explain_match:
            if not self.allow_explain:
                raise SecurityViolationError("EXPLAIN statements are not allowed")
            options = (explain_match.group("options") or "").upper()
            if "ANALYZE" in options and not self.allow_explain_analyze:
                raise SecurityViolationError("EXPLAIN ANALYZE is not allowed")
            if re.match(r"^ANALYZE\b", explain_match.group("inner"), re.IGNORECASE):
                if not self.allow_explain_analyze:
                    raise SecurityViolationError("EXPLAIN ANALYZE is not allowed")
                inner_sql = re.sub(
                    r"^ANALYZE\b", "", explain_match.group("inner"), count=1, flags=re.IGNORECASE
                ).lstrip()
            else:
                inner_sql = explain_match.group("inner")
            # Validate the explained statement itself. This prevents EXPLAIN
            # DELETE/UPDATE from bypassing the read-only and object policies.
            self._validate_parsed_statement(inner_sql)
            return None

        self._validate_parsed_statement(raw_sql)
        return None

    def _validate_parsed_statement(self, sql: str) -> None:
        """Parse and validate one non-EXPLAIN statement."""
        try:
            parsed = sqlglot.parse(sql, read="postgres")
        except Exception as e:
            raise SQLParseError(f"Failed to parse SQL: {e}") from e

        # Check for multiple statements
        if len(parsed) > 1:
            raise SecurityViolationError(
                "Multiple statements not allowed. Only single SELECT queries are permitted."
            )

        if not parsed:
            raise SQLParseError("No valid SQL statement found")

        statement = parsed[0]

        # Check for null or empty statement (e.g., comment-only SQL)
        if statement is None or isinstance(statement, type(None)):
            raise SQLParseError("No valid SQL statement found")

        # EXPLAIN is handled before parsing in ``validate_or_raise`` so that
        # its inner query can be validated recursively and safely.
        if isinstance(statement, exp.Command):
            cmd_name = str(statement.this).upper() if statement.this else ""
            raise SecurityViolationError(
                f"Command '{cmd_name}' is not allowed. Only SELECT queries are permitted."
            )

        # Handle CTE (WITH) statements - extract the main query
        if isinstance(statement, exp.With):
            # WITH statements are allowed, but we need to validate the main query
            if statement.this:
                main_query = statement.this
            else:
                raise SQLParseError("WITH statement has no main query")
        else:
            main_query = statement

        # Perform security checks
        if error := self._check_statement_type(main_query):
            raise SecurityViolationError(error)

        if not self.config.allow_write_operations and any(
            isinstance(node, (exp.Insert, exp.Update, exp.Delete)) for node in statement.walk()
        ):
            raise SecurityViolationError(
                "Write operations in queries are not allowed by the security configuration"
            )

        if error := self._check_dangerous_functions(statement):
            raise SecurityViolationError(error)

        if error := self._check_blocked_tables(statement):
            raise SecurityViolationError(error)

        if error := self._check_allowed_schemas(statement):
            raise SecurityViolationError(error)

        if error := self._check_blocked_schemas(statement):
            raise SecurityViolationError(error)

        if error := self._check_allowed_tables(statement):
            raise SecurityViolationError(error)

        if error := self._check_blocked_columns(statement):
            raise SecurityViolationError(error)

        if error := self._check_allowed_columns(statement):
            raise SecurityViolationError(error)

        if error := self._check_subquery_safety(statement):
            raise SecurityViolationError(error)

    def _check_statement_type(self, statement: exp.Expr) -> str | None:
        """Check if statement type is allowed.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Check for forbidden statement types
        for forbidden_type in self.FORBIDDEN_STATEMENT_TYPES:
            if isinstance(statement, forbidden_type):
                if self.config.allow_write_operations and isinstance(
                    statement, (exp.Insert, exp.Update, exp.Delete)
                ):
                    return None
                stmt_name = forbidden_type.__name__.upper()
                return f"{stmt_name} statements are not allowed. Only SELECT queries are permitted."

        # Ensure statement is an allowed type (SELECT or set operations)
        if not isinstance(statement, tuple(self.ALLOWED_STATEMENT_TYPES)):
            if self.config.allow_write_operations and isinstance(
                statement, (exp.Insert, exp.Update, exp.Delete)
            ):
                return None
            stmt_type = type(statement).__name__
            return f"Statement type {stmt_type} is not allowed. Only SELECT queries are permitted."

        return None

    def _check_dangerous_functions(self, statement: exp.Expr) -> str | None:
        """Check for use of blocked/dangerous functions.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Find all function calls in the query
        for func in statement.find_all(exp.Func):
            func_name = func.name.lower() if func.name else ""

            if func_name in self.blocked_functions:
                return f"Function '{func_name}' is blocked for security reasons"

        return None

    def _check_blocked_tables(self, statement: exp.Expr) -> str | None:
        """Check for access to blocked tables.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        if not self.blocked_tables:
            return None

        # Find all table references
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""
            qualified_name = self._qualified_table_name(table)

            if table_name in self.blocked_tables or qualified_name in self.blocked_tables:
                return f"Access to table '{qualified_name}' is not allowed"

        return None

    @staticmethod
    def _qualified_table_name(table: exp.Table) -> str:
        """Return a normalized schema.table name for a SQLGlot table node."""
        name = (table.name or "").lower()
        schema = (table.db or "").lower()
        return f"{schema}.{name}" if schema else name

    def _check_blocked_schemas(self, statement: exp.Expr) -> str | None:
        if not self.blocked_schemas:
            return None
        search_path = self._safe_search_path_schemas()
        for table in statement.find_all(exp.Table):
            schema = (table.db or "").lower()
            if schema and schema in self.blocked_schemas:
                return f"Access to schema '{schema}' is not allowed"
            if not schema:
                blocked = self.blocked_schemas.intersection(search_path)
                if blocked:
                    return (
                        "Unqualified table access may resolve through blocked schema "
                        f"'{sorted(blocked)[0]}'"
                    )
        return None

    def _check_allowed_schemas(self, statement: exp.Expr) -> str | None:
        if not self.allowed_schemas:
            return None
        search_path = self._safe_search_path_schemas()
        for table in statement.find_all(exp.Table):
            if table.db:
                schema = table.db.lower()
                if schema not in self.allowed_schemas:
                    return f"Schema '{schema}' is not in the allowed schema list"
            elif not search_path or not search_path.issubset(self.allowed_schemas):
                disallowed = search_path - self.allowed_schemas
                return (
                    "Unqualified table access may resolve through a schema outside the "
                    f"allowed schema list: '{sorted(disallowed)[0] if disallowed else 'unknown'}'"
                )
        return None

    def _safe_search_path_schemas(self) -> set[str]:
        """Return the configured, normalized schema search path."""
        return {
            schema.strip().lower()
            for schema in self.config.safe_search_path.split(",")
            if schema.strip()
        }

    def _check_allowed_tables(self, statement: exp.Expr) -> str | None:
        if not self.allowed_tables:
            return None
        for table in statement.find_all(exp.Table):
            table_name = self._qualified_table_name(table)
            short_name = (table.name or "").lower()
            # A short allow-list entry only matches an unqualified reference.
            # It must not authorize an explicitly qualified table in another schema.
            allowed = table_name in self.allowed_tables
            if not table.db and short_name in self.allowed_tables:
                allowed = True
            elif not table.db:
                search_path = self._safe_search_path_schemas()
                possible_names = {f"{schema}.{short_name}" for schema in search_path}
                allowed = bool(possible_names) and possible_names.issubset(self.allowed_tables)
            if not allowed:
                return f"Access to table '{table_name}' is not in the allowed table list"
        return None

    def _check_blocked_columns(self, statement: exp.Expr) -> str | None:
        """Check for access to blocked columns.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        if not self.blocked_columns:
            return None

        aliases = self._table_aliases(statement)
        unaliased_tables = {table.name.lower() for table in statement.find_all(exp.Table)}
        for column in statement.find_all(exp.Column):
            column_name = column.name.lower() if column.name else ""

            # Check for exact match
            if column_name in self.blocked_columns:
                return f"Access to column '{column_name}' is not allowed"

            # Check for qualified column names (table.column)
            if column.table:
                table_name = aliases.get(column.table.lower(), column.table.lower())
                qualified_name = f"{table_name}.{column_name}"
                if qualified_name in self.blocked_columns:
                    return f"Access to column '{qualified_name}' is not allowed"
            else:
                matching_restriction = next(
                    (
                        f"{table_name}.{column_name}"
                        for table_name in unaliased_tables
                        if f"{table_name}.{column_name}" in self.blocked_columns
                    ),
                    None,
                )
                if matching_restriction:
                    return (
                        f"Unqualified column '{column_name}' may access restricted column "
                        f"'{matching_restriction}'"
                    )

        return None

    def _check_allowed_columns(self, statement: exp.Expr) -> str | None:
        if not self.allowed_columns:
            return None
        if any(not isinstance(star.parent, exp.Count) for star in statement.find_all(exp.Star)):
            return "Wildcard column access is not allowed when allowed_columns is configured"
        aliases = self._table_aliases(statement)
        unaliased_tables = {table.name.lower() for table in statement.find_all(exp.Table)}
        for column in statement.find_all(exp.Column):
            column_name = (column.name or "").lower()
            if column.table:
                table_name = aliases.get(column.table.lower(), column.table.lower())
                qualified_names = {column_name, f"{table_name}.{column_name}"}
            elif len(unaliased_tables) == 1:
                table_name = next(iter(unaliased_tables))
                qualified_names = {column_name, f"{table_name}.{column_name}"}
            else:
                qualified_names = {column_name}
            if not qualified_names.intersection(self.allowed_columns):
                qualified_name = next(iter(sorted(qualified_names - {column_name})), column_name)
                return f"Access to column '{qualified_name}' is not in the allowed column list"
        return None

    @staticmethod
    def _table_aliases(statement: exp.Expr) -> dict[str, str]:
        """Map table aliases to underlying table names for column policies."""
        aliases: dict[str, str] = {}
        for table in statement.find_all(exp.Table):
            table_name = (table.name or "").lower()
            if not table_name:
                continue
            aliases[table_name] = table_name
            if table.alias:
                aliases[table.alias.lower()] = table_name
        return aliases

    def _check_subquery_safety(self, statement: exp.Expr) -> str | None:
        """Check that all subqueries only contain SELECT statements.

        Args:
            statement: Parsed SQL statement.

        Returns:
            Error message if check fails, None otherwise.
        """
        # Find all subqueries
        for subquery in statement.find_all(exp.Subquery):
            if subquery.this:
                inner_stmt = subquery.this

                # Check if the inner statement is a forbidden type
                for forbidden_type in self.FORBIDDEN_STATEMENT_TYPES:
                    if isinstance(inner_stmt, forbidden_type):
                        stmt_name = forbidden_type.__name__.upper()
                        return f"{stmt_name} statements in subqueries are not allowed"

                # Ensure it's a SELECT
                if not isinstance(inner_stmt, (exp.Select, exp.With)):
                    return "Subqueries must contain only SELECT statements"

        return None

    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL query to a canonical form.

        This removes extra whitespace, standardizes formatting, and makes
        queries easier to compare or cache.

        Args:
            sql: SQL query string to normalize.

        Returns:
            Normalized SQL string.

        Raises:
            SQLParseError: If SQL cannot be parsed.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
            # Generate normalized SQL
            return parsed.sql(dialect="postgres", pretty=False)
        except Exception as e:
            raise SQLParseError(f"Failed to normalize SQL: {e}") from e

    def extract_tables(self, sql: str) -> list[str]:
        """Extract all table names referenced in the SQL query.

        Args:
            sql: SQL query string.

        Returns:
            List of table names (in lowercase).

        Raises:
            SQLParseError: If SQL cannot be parsed.
        """
        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
            tables = []

            for table in parsed.find_all(exp.Table):
                if table.name:
                    tables.append(table.name.lower())

            return sorted(set(tables))
        except Exception as e:
            raise SQLParseError(f"Failed to extract tables: {e}") from e
