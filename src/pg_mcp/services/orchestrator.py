"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic, error handling, and request tracking.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ResultValidationResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with error feedback, circuit breaker
    pattern for fault tolerance, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executor=executor,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        sql_executors: dict[str, SQLExecutor] | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executor: SQL execution service.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executor = sql_executor
        if sql_executors is not None:
            self.sql_executors = dict(sql_executors)
        elif sql_executor is not None and len(pools) == 1:
            # Preserve the legacy single-database constructor. A single
            # executor must never be reused for several configured databases.
            self.sql_executors = {next(iter(pools)): sql_executor}
        else:
            self.sql_executors = {}
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query under a request-scoped trace and record metrics."""
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        async with request_context(request_id):
            response = await self._execute_query(request, request_id)

        if self.metrics is not None:
            database_label = self._metric_database_label(request.database)
            self.metrics.increment_query_request(
                "success" if response.success else "error", database_label
            )
            self.metrics.query_duration.observe(time.perf_counter() - started)
        return response

    def _metric_database_label(self, requested: str | None) -> str:
        """Use only configured database names as metric labels."""
        if requested is not None:
            return requested if requested in self.pools else "unknown"
        if len(self.pools) == 1:
            return next(iter(self.pools))
        return "unspecified"

    async def _execute_query(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracking
        2. Resolve and validate database name
        3. Load schema from cache
        4. Generate and validate SQL with retry logic
        5. Execute SQL (if return_type == RESULT)
        6. Validate results (optional)
        7. Return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        if len(request.question) > self.validation_config.max_question_length:
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.QUESTION_TOO_LONG.value,
                    message=(
                        "Question exceeds the configured maximum length of "
                        f"{self.validation_config.max_question_length} characters"
                    ),
                    details={"max_length": self.validation_config.max_question_length},
                ),
                confidence=0,
                tokens_used=None,
            )

        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                response = QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )
                return response

            # Step 5: Execute SQL
            logger.debug("Executing SQL", extra={"request_id": request_id})
            start_time = self._get_current_time_ms()
            executor = self.sql_executors.get(database_name)
            if executor is None:
                raise DatabaseError(
                    message=f"No SQL executor available for database '{database_name}'",
                    details={"database": database_name},
                )
            # The public MCP tool holds the query limiter for the complete
            # request; do not acquire the same semaphore a second time here.
            db_started = time.perf_counter()
            try:
                results, total_count = await self._execute_sql_with_retry(
                    executor, generated_sql, request_id
                )
            finally:
                if self.metrics is not None:
                    self.metrics.observe_db_query_duration(time.perf_counter() - db_started)

            execution_time_ms = self._get_current_time_ms() - start_time
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            response = QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )
            return response

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            response = QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
            return response
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            response = QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )
            return response

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM
        3. Validates the generated SQL
        4. On validation failure, retries with error feedback
        5. Records success/failure to circuit breaker

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL
                llm_call_started = False
                llm_started = 0.0
                try:
                    if self.rate_limiter is not None:
                        async with self.rate_limiter.for_llm(
                            timeout=self.resilience_config.rate_limit_timeout
                        ):
                            llm_started = time.perf_counter()
                            llm_call_started = True
                            if self.metrics is not None:
                                self.metrics.increment_llm_call("generate_sql")
                            generated_sql = await self.sql_generator.generate(
                                question=question,
                                schema=schema,
                                previous_attempt=previous_sql,
                                error_feedback=error_feedback,
                            )
                    else:
                        llm_started = time.perf_counter()
                        llm_call_started = True
                        if self.metrics is not None:
                            self.metrics.increment_llm_call("generate_sql")
                        generated_sql = await self.sql_generator.generate(
                            question=question,
                            schema=schema,
                            previous_attempt=previous_sql,
                            error_feedback=error_feedback,
                        )
                except (LLMTimeoutError, LLMUnavailableError, LLMError) as generation_error:
                    if self._is_retryable_llm_error(generation_error) and attempt < max_retries:
                        self.circuit_breaker.record_failure()
                        delay = self._retry_delay(attempt)
                        if self.metrics is not None:
                            self.metrics.increment_retry_attempt("sql_generation")
                        logger.warning(
                            "Transient LLM failure; retrying with backoff",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "retry_in_seconds": delay,
                                "error_type": type(generation_error).__name__,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue
                    self.circuit_breaker.record_failure()
                    raise
                finally:
                    if self.metrics is not None and llm_call_started:
                        self.metrics.observe_llm_latency(
                            "generate_sql", time.perf_counter() - llm_started
                        )

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if self.metrics is not None:
                        self.metrics.increment_sql_rejected(type(validation_error).__name__.lower())
                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        if self.metrics is not None:
                            self.metrics.increment_retry_attempt("sql_validation")
                        await asyncio.sleep(self._retry_delay(attempt))
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except RateLimitExceededError:
                if self.metrics is not None:
                    self.metrics.increment_rate_limit_rejection()
                raise
            except (LLMError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    def _retry_delay(self, attempt: int) -> float:
        """Calculate exponential retry delay from active resilience settings."""
        return min(
            self.resilience_config.retry_delay * (self.resilience_config.backoff_factor**attempt),
            60.0,
        )

    async def _execute_sql_with_retry(
        self, executor: SQLExecutor, sql: str, request_id: str
    ) -> tuple[list[dict[str, Any]], int]:
        """Retry transient database failures with configured exponential backoff."""
        for attempt in range(self.resilience_config.max_retries + 1):
            try:
                return await executor.execute(sql)
            except DatabaseError as error:
                if (
                    not self._is_retryable_database_error(error)
                    or attempt >= self.resilience_config.max_retries
                ):
                    raise
                delay = self._retry_delay(attempt)
                if self.metrics is not None:
                    self.metrics.increment_retry_attempt("database")
                logger.warning(
                    "Transient database failure; retrying with backoff",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "retry_in_seconds": delay,
                        "database_error_code": error.details.get("error_code"),
                    },
                )
                await asyncio.sleep(delay)
        raise AssertionError("Database retry loop exited unexpectedly")

    @staticmethod
    def _is_retryable_database_error(error: DatabaseError) -> bool:
        """Retry connection, serialization, deadlock, and temporary resource errors."""
        code = str(error.details.get("error_code") or "")
        return code.startswith(("08", "40", "53")) or code in {
            "55P03",  # lock_not_available
            "57P01",  # admin_shutdown
            "57P02",  # crash_shutdown
            "57P03",  # cannot_connect_now
        }

    @staticmethod
    def _is_retryable_llm_error(error: LLMError) -> bool:
        """Retry timeouts and transient connectivity/rate-limit failures only."""
        if isinstance(error, LLMTimeoutError):
            return True
        message = f"{error.message} {error.details}".lower()
        return any(
            marker in message
            for marker in ("rate limit", "429", "temporarily", "connection", "network", "timeout")
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self._validate_results_with_retry(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
                request_id=request_id,
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except RateLimitExceededError as e:
            if self.metrics is not None:
                self.metrics.increment_rate_limit_rejection()
            logger.warning(
                "Result validation rate limit reached; returning query result without validation",
                extra={"request_id": request_id, "error": str(e)},
            )
            return 100
        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    async def _validate_results_with_retry(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> ResultValidationResult:
        """Run result validation under the LLM limiter with transient retries."""
        for attempt in range(self.resilience_config.max_retries + 1):
            started: float | None = None
            try:
                if self.rate_limiter is not None:
                    async with self.rate_limiter.for_llm(
                        timeout=self.resilience_config.rate_limit_timeout
                    ):
                        started = time.perf_counter()
                        if self.metrics is not None:
                            self.metrics.increment_llm_call("validate_result")
                        result = await self.result_validator.validate(
                            question=question,
                            sql=sql,
                            results=results,
                            row_count=row_count,
                        )
                else:
                    started = time.perf_counter()
                    if self.metrics is not None:
                        self.metrics.increment_llm_call("validate_result")
                    result = await self.result_validator.validate(
                        question=question,
                        sql=sql,
                        results=results,
                        row_count=row_count,
                    )
                return result
            except LLMError as error:
                if (
                    not self._is_retryable_llm_error(error)
                    or attempt >= self.resilience_config.max_retries
                ):
                    raise
                delay = self._retry_delay(attempt)
                if self.metrics is not None:
                    self.metrics.increment_retry_attempt("result_validation")
                logger.warning(
                    "Transient result validation failure; retrying with backoff",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "retry_in_seconds": delay,
                        "error_type": type(error).__name__,
                    },
                )
                await asyncio.sleep(delay)
            finally:
                if started is not None and self.metrics is not None:
                    self.metrics.observe_llm_latency(
                        "validate_result", time.perf_counter() - started
                    )
        raise AssertionError("Result validation retry loop exited unexpectedly")

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        import time

        return time.time() * 1000
