from clickhouse_driver import Client
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
import sys
import subprocess
import os
from queue import Queue
import threading
import time
import uuid
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)


class ClickHouseDB:
    def __init__(
        self,
        host="localhost",
        port=9000,
        database="crypto_bot",
        batch_size=1000,
        flush_interval=5,
    ):
        try:
            # Create a connection pool with locks
            self.connection_pool = []
            self.connection_locks = []
            self.pool_size = 3  # Number of connections in the pool
            for _ in range(self.pool_size):
                client = Client(host=host, port=port, database=database)
                self.connection_pool.append(client)
                self.connection_locks.append(Lock())

            self._init_tables()

            # Initialize batch processing
            self.batch_size = batch_size
            self.flush_interval = flush_interval
            self.trades_queue = Queue()
            self.iterations_queue = Queue()
            self.sub_iterations_queue = Queue()

            # Start background threads for batch processing
            self.running = True
            self.trades_thread = threading.Thread(target=self._process_trades_batch)
            self.iterations_thread = threading.Thread(
                target=self._process_iterations_batch
            )
            self.sub_iterations_thread = threading.Thread(
                target=self._process_sub_iterations_batch
            )

            self.trades_thread.start()
            self.iterations_thread.start()
            self.sub_iterations_thread.start()

        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            logger.error("Please make sure ClickHouse is properly configured:")
            logger.error("1. Check if ClickHouse is installed: clickhouse --version")
            logger.error("2. Check if server is running: clickhouse server")
            logger.error("3. Check if ports 8123 and 9000 are available")
            self.client = None

    def _get_connection(self):
        """Get a connection from the pool using round-robin with lock"""
        if not self.connection_pool:
            return None, None
        
        # Get connection index using thread ID
        conn_index = threading.get_ident() % len(self.connection_pool)
        return self.connection_pool[conn_index], self.connection_locks[conn_index]

    def _init_tables(self):
        if not self.connection_pool:
            return

        try:
            client, _ = self._get_connection()
            # Create database if it doesn't exist
            client.execute("CREATE DATABASE IF NOT EXISTS crypto_bot")
            client.execute("USE crypto_bot")

            # Create trades table with async insert settings
            client.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id UUID DEFAULT generateUUIDv4(),
                    iteration_id UUID,
                    sub_iteration_id UUID,
                    symbol String,
                    interval String,
                    trade_type Enum('buy' = 1, 'sell' = 2, 'sell_end' = 3),
                    entry_timestamp DateTime,
                    exit_timestamp DateTime,
                    entry_index UInt32,
                    exit_index UInt32,
                    entry_price Float64,
                    exit_price Float64,
                    profit_loss Float64,
                    trade_duration UInt32,
                    entry_signal String,
                    exit_signal String,
                    created_at DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (entry_timestamp, iteration_id, sub_iteration_id)
                SETTINGS async_insert = 1, wait_for_async_insert = 0
            """
            )

            # Create iterations table with async insert settings
            client.execute(
                """
                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id UUID DEFAULT generateUUIDv4(),
                    iteration_number UInt32,
                    weights Array(Float64),
                    fitness_score Float64,
                    total_trades UInt32,
                    win_rate Float64,
                    total_revenue Float64,
                    max_drawdown Float64,
                    avg_profit Float64,
                    avg_loss Float64,
                    profit_factor Float64,
                    avg_trade_duration Float64,
                    created_at DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (iteration_number)
                SETTINGS async_insert = 1, wait_for_async_insert = 0
            """
            )

            # Create sub_iterations table with async insert settings
            client.execute(
                """
                CREATE TABLE IF NOT EXISTS sub_iterations (
                    sub_iteration_id UUID DEFAULT generateUUIDv4(),
                    iteration_id UUID,
                    symbol String,
                    interval String,
                    candles UInt32,
                    window UInt32,
                    initial_balance Float64,
                    final_balance Float64,
                    total_trades UInt32,
                    win_rate Float64,
                    revenue Float64,
                    created_at DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (iteration_id, created_at)
                SETTINGS async_insert = 1, wait_for_async_insert = 0
            """
            )
        except Exception as e:
            logger.error(f"Failed to initialize tables: {str(e)}")
            self.connection_pool = []

    def _ensure_required_fields(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> Dict[str, Any]:
        """Ensure all required fields are present in the data"""
        current_time = datetime.now()

        # Ensure created_at is present
        if "created_at" not in data:
            data["created_at"] = current_time

        # Ensure other required fields are present
        for field in required_fields:
            if field not in data:
                if field == "trade_id":
                    data[field] = str(uuid.uuid4())
                elif field == "iteration_id":
                    data[field] = str(uuid.uuid4())
                elif field == "sub_iteration_id":
                    data[field] = str(uuid.uuid4())

        return data

    def _process_trades_batch(self):
        while self.running:
            batch = []
            try:
                # Collect items up to batch_size or until timeout
                start_time = time.time()
                while (
                    len(batch) < self.batch_size
                    and time.time() - start_time < self.flush_interval
                ):
                    try:
                        item = self.trades_queue.get(timeout=0.1)
                        item = self._ensure_required_fields(
                            item, ["trade_id", "created_at"]
                        )
                        batch.append(item)
                    except:
                        break

                if batch:
                    client, lock = self._get_connection()
                    if client:
                        with lock:
                            client.execute("INSERT INTO trades VALUES", batch)
            except Exception as e:
                logger.error(f"Error processing trades batch: {str(e)}")

    def _process_iterations_batch(self):
        while self.running:
            batch = []
            try:
                # Collect items up to batch_size or until timeout
                start_time = time.time()
                while (
                    len(batch) < self.batch_size
                    and time.time() - start_time < self.flush_interval
                ):
                    try:
                        item = self.iterations_queue.get(timeout=0.1)
                        item = self._ensure_required_fields(
                            item, ["iteration_id", "created_at"]
                        )
                        batch.append(item)
                    except:
                        break

                if batch:
                    client, lock = self._get_connection()
                    if client:
                        with lock:
                            client.execute("INSERT INTO iterations VALUES", batch)
            except Exception as e:
                logger.error(f"Error processing iterations batch: {str(e)}")

    def _process_sub_iterations_batch(self):
        while self.running:
            batch = []
            try:
                # Collect items up to batch_size or until timeout
                start_time = time.time()
                while (
                    len(batch) < self.batch_size
                    and time.time() - start_time < self.flush_interval
                ):
                    try:
                        item = self.sub_iterations_queue.get(timeout=0.1)
                        item = self._ensure_required_fields(
                            item, ["sub_iteration_id", "created_at"]
                        )
                        batch.append(item)
                    except:
                        break

                if batch:
                    client, lock = self._get_connection()
                    if client:
                        with lock:
                            client.execute("INSERT INTO sub_iterations VALUES", batch)
            except Exception as e:
                logger.error(f"Error processing sub-iterations batch: {str(e)}")

    def insert_trade(self, trade_data: Dict[str, Any]) -> Optional[str]:
        if not self.connection_pool:
            logger.warning("Database not connected, skipping trade insertion")
            return None

        try:
            trade_data = self._ensure_required_fields(
                trade_data, ["trade_id", "created_at"]
            )
            self.trades_queue.put(trade_data)
            return trade_data["trade_id"]
        except Exception as e:
            logger.error(f"Failed to queue trade: {str(e)}")
            return None

    def insert_iteration(self, iteration_data: Dict[str, Any]) -> Optional[str]:
        if not self.connection_pool:
            logger.warning("Database not connected, skipping iteration insertion")
            return None

        try:
            iteration_data = self._ensure_required_fields(
                iteration_data, ["iteration_id", "created_at"]
            )
            self.iterations_queue.put(iteration_data)
            return iteration_data["iteration_id"]
        except Exception as e:
            logger.error(f"Failed to queue iteration: {str(e)}")
            return None

    def insert_sub_iteration(self, sub_iteration_data: Dict[str, Any]) -> Optional[str]:
        if not self.connection_pool:
            logger.warning("Database not connected, skipping sub-iteration insertion")
            return None

        try:
            sub_iteration_data = self._ensure_required_fields(
                sub_iteration_data, ["sub_iteration_id", "created_at"]
            )
            self.sub_iterations_queue.put(sub_iteration_data)
            return sub_iteration_data["sub_iteration_id"]
        except Exception as e:
            logger.error(f"Failed to queue sub-iteration: {str(e)}")
            return None

    def get_trades_by_iteration(self, iteration_id: str) -> List[Dict[str, Any]]:
        if not self.connection_pool:
            logger.warning("Database not connected, returning empty trade list")
            return []

        try:
            client, _ = self._get_connection()
            if not client:
                return []
            query = """
                SELECT * FROM trades
                WHERE iteration_id = %(iteration_id)s
                ORDER BY entry_timestamp
            """
            return client.execute(query, {"iteration_id": iteration_id})
        except Exception as e:
            logger.error(f"Failed to get trades: {str(e)}")
            return []

    def get_sub_iterations_by_iteration(
        self, iteration_id: str
    ) -> List[Dict[str, Any]]:
        if not self.connection_pool:
            logger.warning("Database not connected, returning empty sub-iteration list")
            return []

        try:
            client, _ = self._get_connection()
            if not client:
                return []
            query = """
                SELECT * FROM sub_iterations
                WHERE iteration_id = %(iteration_id)s
                ORDER BY created_at
            """
            return client.execute(query, {"iteration_id": iteration_id})
        except Exception as e:
            logger.error(f"Failed to get sub-iterations: {str(e)}")
            return []

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Execute a custom SQL query and return the results.

        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Optional parameters for the query

        Returns:
            Union[List[Dict[str, Any]], str]: Query results as a list of dictionaries or error message
        """
        if not self.connection_pool:
            return "Database not connected"

        try:
            client, lock = self._get_connection()
            if not client:
                return "No available database connection"

            # Execute the query with lock
            with lock:
                result = client.execute(query, params or {})

                # If the query includes FORMAT clause, return raw result
                if "FORMAT" in query.upper():
                    return result

                # Get column names from the query
                try:
                    # Try to get column names from the result
                    if result and len(result) > 0:
                        column_names = [f"column_{i}" for i in range(len(result[0]))]
                    else:
                        return "No results returned"

                    # Convert result to list of dictionaries
                    formatted_result = []
                    for row in result:
                        formatted_result.append(dict(zip(column_names, row)))

                    return formatted_result
                except Exception as e:
                    logger.error(f"Error formatting results: {str(e)}")
                    return result

        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def execute_query_to_dataframe(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Union[pd.DataFrame, str]:
        """
        Execute a custom SQL query and return the results as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute
            params (Optional[Dict[str, Any]]): Optional parameters for the query

        Returns:
            Union[pd.DataFrame, str]: Query results as a pandas DataFrame or error message
        """
        if not self.connection_pool:
            return "Database not connected"

        try:
            client, lock = self._get_connection()
            if not client:
                return "No available database connection"

            # Execute the query with lock
            with lock:
                result = client.execute(query, params or {})

                # Get column names from the query
                column_names = client.execute(f"DESCRIBE TABLE ({query})")
                if not column_names:
                    # If DESCRIBE fails, try to get column names from the result
                    if result and len(result) > 0:
                        column_names = [f"column_{i}" for i in range(len(result[0]))]
                    else:
                        return "No results returned"

                # Convert to DataFrame
                df = pd.DataFrame(result, columns=column_names)
                return df

        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_table_schema(self, table_name: str) -> Union[List[Dict[str, Any]], str]:
        """
        Get the schema of a specific table.

        Args:
            table_name (str): Name of the table

        Returns:
            Union[List[Dict[str, Any]], str]: Table schema or error message
        """
        if not self.connection_pool:
            return "Database not connected"

        try:
            client, _ = self._get_connection()
            if not client:
                return "No available database connection"

            # Get table schema
            schema = client.execute(f"DESCRIBE TABLE {table_name}")

            # Format the result
            formatted_schema = []
            for column in schema:
                formatted_schema.append(
                    {
                        "name": column[0],
                        "type": column[1],
                        "default": column[2],
                        "compression_codec": column[3],
                        "ttl": column[4],
                    }
                )

            return formatted_schema

        except Exception as e:
            error_msg = f"Error getting table schema: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_available_tables(self) -> Union[List[str], str]:
        """
        Get a list of all available tables in the database.

        Returns:
            Union[List[str], str]: List of table names or error message
        """
        if not self.connection_pool:
            return "Database not connected"

        try:
            client, _ = self._get_connection()
            if not client:
                return "No available database connection"

            # Get list of tables
            tables = client.execute("SHOW TABLES")
            return [table[0] for table in tables]

        except Exception as e:
            error_msg = f"Error getting available tables: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.running = False
        if hasattr(self, "trades_thread"):
            self.trades_thread.join()
        if hasattr(self, "iterations_thread"):
            self.iterations_thread.join()
        if hasattr(self, "sub_iterations_thread"):
            self.sub_iterations_thread.join()
