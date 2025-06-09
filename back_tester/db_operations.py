from clickhouse_driver import Client
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import logging
import sys
import subprocess
import os
from queue import Queue, Empty
import threading
import time
import uuid
import pandas as pd
from threading import Lock
import json
import numpy as np

logger = logging.getLogger(__name__)


class ClickHouseDB:
    def __init__(
        self,
        host="localhost",
        port=9000,
        database="crypto_bot",
        batch_size=500,
        flush_interval=2,
    ):
        try:
            # Create a connection pool with locks
            self.connection_pool = []
            self.connection_locks = []
            self.pool_size = 3  # Number of connections in the pool
            self.host = host
            self.port = port
            self.database = database

            # Initialize connection pool
            self._init_connection_pool()

            self._init_tables()

            # Initialize batch processing
            self.batch_size = batch_size
            self.flush_interval = flush_interval
            self.trades_queue = Queue()
            self.iterations_queue = Queue()
            self.sub_iterations_queue = Queue()

            # Start background threads for batch processing
            self.running = True
            self.trades_thread = threading.Thread(
                target=self._process_trades_batch, daemon=True
            )
            self.iterations_thread = threading.Thread(
                target=self._process_iterations_batch, daemon=True
            )
            self.sub_iterations_thread = threading.Thread(
                target=self._process_sub_iterations_batch, daemon=True
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
            self.connection_pool = []

    def _init_connection_pool(self):
        """Initialize connection pool with error handling"""
        for i in range(self.pool_size):
            try:
                client = Client(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    send_receive_timeout=60,
                    connect_timeout=10,
                    sync_request_timeout=60,
                    compression=False,  # Disable compression to avoid data corruption
                )
                # Test connection
                client.execute("SELECT 1")
                self.connection_pool.append(client)
                self.connection_locks.append(Lock())
            except Exception as e:
                logger.error(f"Failed to create connection {i}: {str(e)}")

    def _get_connection(self):
        """Get a connection from the pool using round-robin with lock"""
        if not self.connection_pool:
            return None, None

        # Get connection index using thread ID
        conn_index = threading.get_ident() % len(self.connection_pool)
        client = self.connection_pool[conn_index]
        lock = self.connection_locks[conn_index]

        # Test if connection is alive
        try:
            with lock:
                client.execute("SELECT 1")
        except Exception:
            # Reconnect if connection is dead
            try:
                client.disconnect()
            except:
                pass

            try:
                new_client = Client(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    send_receive_timeout=60,
                    connect_timeout=10,
                    sync_request_timeout=60,
                    compression=False,
                )
                new_client.execute("SELECT 1")
                self.connection_pool[conn_index] = new_client
                client = new_client
            except Exception as e:
                logger.error(f"Failed to reconnect: {str(e)}")
                return None, None

        return client, lock

    def _init_tables(self):
        if not self.connection_pool:
            return

        try:
            client, lock = self._get_connection()
            if not client:
                return

            with lock:
                # Create database if it doesn't exist
                client.execute("CREATE DATABASE IF NOT EXISTS crypto_bot")
                client.execute("USE crypto_bot")

                # Create trades table with risk management fields
                client.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id UUID DEFAULT generateUUIDv4(),
                        iteration_id UUID,
                        sub_iteration_id UUID,
                        symbol String,
                        interval String,
                        trade_type String,
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
                        risk_reward_ratio Float64,
                        position_size Float64,
                        stop_loss Float64,
                        take_profit_1 Float64,
                        take_profit_2 Float64,
                        take_profit_3 Float64,
                        risk_percentage Float64,
                        amount_traded Float64,
                        parent_trade_id Nullable(UUID),
                        created_at DateTime DEFAULT now()
                    ) ENGINE = MergeTree()
                    ORDER BY (entry_timestamp, iteration_id, sub_iteration_id)
                    """
                )

                # Create iterations table with risk management metrics
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
                        tp1_hits UInt32,
                        tp2_hits UInt32,
                        tp3_hits UInt32,
                        stop_loss_hits UInt32,
                        avg_risk_reward Float64,
                        avg_position_size Float64,
                        risk_percentage Float64,
                        created_at DateTime DEFAULT now()
                    ) ENGINE = MergeTree()
                    ORDER BY (iteration_number)
                    """
                )

                # Create sub_iterations table with risk management fields
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
                        risk_percentage Float64,
                        avg_risk_reward Float64,
                        tp_success_rate Float64,
                        created_at DateTime DEFAULT now()
                    ) ENGINE = MergeTree()
                    ORDER BY (iteration_id, created_at)
                    """
                )
        except Exception as e:
            logger.error(f"Failed to initialize tables: {str(e)}")
            self.connection_pool = []

    def _ensure_required_fields(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> Dict[str, Any]:
        """Ensure all required fields are present and have correct types"""
        result = data.copy()
        current_time = datetime.now()

        # Define default values based on field types
        defaults = {
            "created_at": current_time,
            "trade_id": None,
            "iteration_id": None,
            "sub_iteration_id": None,
            "parent_trade_id": None,
            "symbol": "",
            "interval": "",
            "trade_type": "entry",
            "entry_timestamp": current_time,
            "exit_timestamp": current_time,
            "entry_index": 0,
            "exit_index": 0,
            "entry_price": 0.0,
            "exit_price": 0.0,
            "profit_loss": 0.0,
            "trade_duration": 0,
            "entry_signal": "",
            "exit_signal": "",
            "risk_reward_ratio": 0.0,
            "position_size": 0.0,
            "stop_loss": 0.0,
            "take_profit_1": 0.0,
            "take_profit_2": 0.0,
            "take_profit_3": 0.0,
            "risk_percentage": 0.0,
            "amount_traded": 0.0,
            "tp1_hits": 0,
            "tp2_hits": 0,
            "tp3_hits": 0,
            "stop_loss_hits": 0,
            "avg_risk_reward": 0.0,
            "avg_position_size": 0.0,
            "tp_success_rate": 0.0,
            "weights": [],
            "iteration_number": 0,
            "fitness_score": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_revenue": 0.0,
            "max_drawdown": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": 0.0,
            "candles": 0,
            "window": 0,
            "initial_balance": 0.0,
            "final_balance": 0.0,
            "revenue": 0.0,
        }

        # Ensure all required fields are present with correct types
        for field in required_fields:
            if field not in result or result[field] is None:
                if field in defaults:
                    result[field] = defaults[field]
                else:
                    raise ValueError(
                        f"Required field {field} is missing and has no default value"
                    )

            # Convert numeric fields to float
            if field in [
                "entry_price",
                "exit_price",
                "profit_loss",
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
                "take_profit_3",
                "risk_reward_ratio",
                "position_size",
                "risk_percentage",
                "avg_risk_reward",
                "avg_position_size",
                "tp_success_rate",
                "amount_traded",
                "fitness_score",
                "win_rate",
                "total_revenue",
                "max_drawdown",
                "avg_profit",
                "avg_loss",
                "profit_factor",
                "avg_trade_duration",
                "initial_balance",
                "final_balance",
                "revenue",
            ]:
                try:
                    result[field] = float(result[field])
                    # Check for NaN or Inf
                    if pd.isna(result[field]) or np.isinf(result[field]):
                        result[field] = 0.0
                except (ValueError, TypeError):
                    result[field] = 0.0

            # Convert integer fields
            if field in [
                "entry_index",
                "exit_index",
                "trade_duration",
                "tp1_hits",
                "tp2_hits",
                "tp3_hits",
                "stop_loss_hits",
                "iteration_number",
                "total_trades",
                "candles",
                "window",
            ]:
                try:
                    result[field] = int(result[field])
                except (ValueError, TypeError):
                    result[field] = 0

            # Ensure UUIDs are strings
            if field in [
                "trade_id",
                "iteration_id",
                "sub_iteration_id",
                "parent_trade_id",
            ]:
                if field == "parent_trade_id" and (
                    result[field] is None
                    or result[field] == ""
                    or result[field] == "None"
                ):
                    result[field] = None  # parent_trade_id can be null
                elif (
                    result[field] is None
                    or result[field] == ""
                    or result[field] == "None"
                ):
                    # Generate new UUID if missing
                    result[field] = str(uuid.uuid4())
                elif not isinstance(result[field], str):
                    result[field] = str(result[field])

            # Ensure timestamps are datetime objects
            if field in ["entry_timestamp", "exit_timestamp", "created_at"]:
                if not isinstance(result[field], datetime):
                    result[field] = current_time

            # Ensure arrays are lists
            if field == "weights":
                if not isinstance(result[field], list):
                    result[field] = []

        return result

    def _validate_batch_data(
        self, batch: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        """Validate and clean batch data before insertion"""
        validated_batch = []

        # Default values for missing fields
        defaults = {
            "created_at": datetime.now(),
            "trade_id": str(uuid.uuid4()),
            "iteration_id": str(uuid.uuid4()),
            "sub_iteration_id": str(uuid.uuid4()),
            "parent_trade_id": None,
            "symbol": "",
            "interval": "",
            "trade_type": "entry",
            "entry_timestamp": datetime.now(),
            "exit_timestamp": datetime.now(),
            "entry_index": 0,
            "exit_index": 0,
            "entry_price": 0.0,
            "exit_price": 0.0,
            "profit_loss": 0.0,
            "trade_duration": 0,
            "entry_signal": "",
            "exit_signal": "",
            "risk_reward_ratio": 0.0,
            "position_size": 0.0,
            "stop_loss": 0.0,
            "take_profit_1": 0.0,
            "take_profit_2": 0.0,
            "take_profit_3": 0.0,
            "risk_percentage": 0.0,
            "amount_traded": 0.0,
        }

        for item in batch:
            try:
                # Deep copy to avoid modifying original
                clean_item = {}

                for key, value in item.items():
                    # Handle None values
                    if value is None:
                        if key == "parent_trade_id":
                            clean_item[key] = None
                        else:
                            clean_item[key] = defaults.get(key, 0)
                    # Handle NaN and Inf values
                    elif isinstance(value, float):
                        if pd.isna(value) or np.isinf(value):
                            clean_item[key] = 0.0
                        else:
                            clean_item[key] = value
                    # Handle empty strings for UUIDs
                    elif key in ["trade_id", "iteration_id", "sub_iteration_id"] and (
                        value == "" or value == "None"
                    ):
                        clean_item[key] = str(uuid.uuid4())
                    else:
                        clean_item[key] = value

                validated_batch.append(clean_item)
            except Exception as e:
                logger.warning(f"Skipping invalid item in {table_name}: {str(e)}")

        return validated_batch

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
                        # Ensure all required fields are present and properly formatted
                        item = self._ensure_required_fields(
                            item,
                            [
                                "trade_id",
                                "iteration_id",
                                "sub_iteration_id",
                                "symbol",
                                "interval",
                                "trade_type",
                                "entry_timestamp",
                                "exit_timestamp",
                                "entry_index",
                                "exit_index",
                                "entry_price",
                                "exit_price",
                                "profit_loss",
                                "trade_duration",
                                "entry_signal",
                                "exit_signal",
                                "risk_reward_ratio",
                                "position_size",
                                "stop_loss",
                                "take_profit_1",
                                "take_profit_2",
                                "take_profit_3",
                                "risk_percentage",
                                "amount_traded",
                                "parent_trade_id",
                                "created_at",
                            ],
                        )
                        batch.append(item)
                    except Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing trade item: {str(e)}")

                if batch:
                    # Validate batch data
                    validated_batch = self._validate_batch_data(batch, "trades")

                    if validated_batch:
                        client, lock = self._get_connection()
                        if client:
                            try:
                                with lock:
                                    # Insert in smaller chunks to avoid connection issues
                                    chunk_size = 100
                                    for i in range(0, len(validated_batch), chunk_size):
                                        chunk = validated_batch[i : i + chunk_size]
                                        client.execute(
                                            "INSERT INTO trades VALUES", chunk
                                        )
                                        time.sleep(0.01)  # Small delay between chunks
                            except Exception as e:
                                logger.error(f"Error inserting trades batch: {str(e)}")
                                # Try to reconnect
                                self._init_connection_pool()

            except Exception as e:
                logger.error(f"Error processing trades batch: {str(e)}")
                time.sleep(1)  # Wait before retrying

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
                        # Ensure all required fields are present and properly formatted
                        item = self._ensure_required_fields(
                            item,
                            [
                                "iteration_id",
                                "iteration_number",
                                "weights",
                                "fitness_score",
                                "total_trades",
                                "win_rate",
                                "total_revenue",
                                "max_drawdown",
                                "avg_profit",
                                "avg_loss",
                                "profit_factor",
                                "avg_trade_duration",
                                "tp1_hits",
                                "tp2_hits",
                                "tp3_hits",
                                "stop_loss_hits",
                                "avg_risk_reward",
                                "avg_position_size",
                                "risk_percentage",
                                "created_at",
                            ],
                        )
                        batch.append(item)
                    except Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing iteration item: {str(e)}")

                if batch:
                    # Validate batch data
                    validated_batch = self._validate_batch_data(batch, "iterations")

                    if validated_batch:
                        client, lock = self._get_connection()
                        if client:
                            try:
                                with lock:
                                    client.execute(
                                        "INSERT INTO iterations VALUES", validated_batch
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error inserting iterations batch: {str(e)}"
                                )
                                # Try to reconnect
                                self._init_connection_pool()

            except Exception as e:
                logger.error(f"Error processing iterations batch: {str(e)}")
                time.sleep(1)

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
                        # Ensure all required fields are present and properly formatted
                        item = self._ensure_required_fields(
                            item,
                            [
                                "sub_iteration_id",
                                "iteration_id",
                                "symbol",
                                "interval",
                                "candles",
                                "window",
                                "initial_balance",
                                "final_balance",
                                "total_trades",
                                "win_rate",
                                "revenue",
                                "risk_percentage",
                                "avg_risk_reward",
                                "tp_success_rate",
                                "created_at",
                            ],
                        )
                        batch.append(item)
                    except Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing sub-iteration item: {str(e)}")

                if batch:
                    # Validate batch data
                    validated_batch = self._validate_batch_data(batch, "sub_iterations")

                    if validated_batch:
                        client, lock = self._get_connection()
                        if client:
                            try:
                                with lock:
                                    client.execute(
                                        "INSERT INTO sub_iterations VALUES",
                                        validated_batch,
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error inserting sub-iterations batch: {str(e)}"
                                )
                                # Try to reconnect
                                self._init_connection_pool()

            except Exception as e:
                logger.error(f"Error processing sub-iterations batch: {str(e)}")
                time.sleep(1)

    def insert_trade(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """Insert a trade record into the database"""
        if not self.connection_pool:
            logger.warning("Database not connected, skipping trade insertion")
            return None

        try:
            # Ensure trade_id is a valid UUID
            if "trade_id" not in trade_data or not trade_data["trade_id"]:
                trade_data["trade_id"] = str(uuid.uuid4())

            trade_data = self._ensure_required_fields(
                trade_data,
                [
                    "trade_id",
                    "iteration_id",
                    "sub_iteration_id",
                    "symbol",
                    "interval",
                    "trade_type",
                    "entry_timestamp",
                    "exit_timestamp",
                    "entry_index",
                    "exit_index",
                    "entry_price",
                    "exit_price",
                    "profit_loss",
                    "trade_duration",
                    "entry_signal",
                    "exit_signal",
                    "risk_reward_ratio",
                    "position_size",
                    "stop_loss",
                    "take_profit_1",
                    "take_profit_2",
                    "take_profit_3",
                    "risk_percentage",
                    "amount_traded",
                    "parent_trade_id",
                    "created_at",
                ],
            )
            self.trades_queue.put(trade_data)
            return trade_data["trade_id"]
        except Exception as e:
            logger.error(f"Error inserting trade: {str(e)}")
            return None

    def insert_iteration(self, iteration_data: Dict[str, Any]) -> Optional[str]:
        """Insert an iteration record into the database"""
        if not self.connection_pool:
            logger.warning("Database not connected, skipping iteration insertion")
            return None

        try:
            if (
                "iteration_id" not in iteration_data
                or not iteration_data["iteration_id"]
            ):
                iteration_data["iteration_id"] = str(uuid.uuid4())

            iteration_data = self._ensure_required_fields(
                iteration_data,
                [
                    "iteration_id",
                    "iteration_number",
                    "weights",
                    "fitness_score",
                    "total_trades",
                    "win_rate",
                    "total_revenue",
                    "max_drawdown",
                    "avg_profit",
                    "avg_loss",
                    "profit_factor",
                    "avg_trade_duration",
                    "tp1_hits",
                    "tp2_hits",
                    "tp3_hits",
                    "stop_loss_hits",
                    "avg_risk_reward",
                    "avg_position_size",
                    "risk_percentage",
                    "created_at",
                ],
            )
            self.iterations_queue.put(iteration_data)
            return iteration_data["iteration_id"]
        except Exception as e:
            logger.error(f"Error inserting iteration: {str(e)}")
            return None

    def insert_sub_iteration(self, sub_iteration_data: Dict[str, Any]) -> Optional[str]:
        """Insert a sub-iteration record into the database"""
        if not self.connection_pool:
            logger.warning("Database not connected, skipping sub-iteration insertion")
            return None

        try:
            # Ensure sub_iteration_id is a valid UUID
            if (
                "sub_iteration_id" not in sub_iteration_data
                or not sub_iteration_data["sub_iteration_id"]
            ):
                sub_iteration_data["sub_iteration_id"] = str(uuid.uuid4())

            # Ensure iteration_id is a valid UUID
            if (
                "iteration_id" not in sub_iteration_data
                or not sub_iteration_data["iteration_id"]
            ):
                sub_iteration_data["iteration_id"] = str(uuid.uuid4())

            sub_iteration_data = self._ensure_required_fields(
                sub_iteration_data,
                [
                    "sub_iteration_id",
                    "iteration_id",
                    "symbol",
                    "interval",
                    "candles",
                    "window",
                    "initial_balance",
                    "final_balance",
                    "total_trades",
                    "win_rate",
                    "revenue",
                    "risk_percentage",
                    "avg_risk_reward",
                    "tp_success_rate",
                    "created_at",
                ],
            )
            self.sub_iterations_queue.put(sub_iteration_data)
            return sub_iteration_data["sub_iteration_id"]
        except Exception as e:
            logger.error(f"Error inserting sub-iteration: {str(e)}")
            return None

    def get_trades_by_iteration(self, iteration_id: str) -> List[Dict[str, Any]]:
        if not self.connection_pool:
            logger.warning("Database not connected, returning empty trade list")
            return []

        try:
            client, lock = self._get_connection()
            if not client:
                return []

            with lock:
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
            client, lock = self._get_connection()
            if not client:
                return []

            with lock:
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
                try:
                    column_names = client.execute(f"DESCRIBE ({query})")
                    column_names = [col[0] for col in column_names]
                except:
                    # If DESCRIBE fails, try to get column names from the result
                    if result and len(result) > 0:
                        column_names = [f"column_{i}" for i in range(len(result[0]))]
                    else:
                        return pd.DataFrame()  # Return empty DataFrame

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
            client, lock = self._get_connection()
            if not client:
                return "No available database connection"

            with lock:
                # Get table schema
                schema = client.execute(f"DESCRIBE TABLE {table_name}")

                # Format the result
                formatted_schema = []
                for column in schema:
                    formatted_schema.append(
                        {
                            "name": column[0],
                            "type": column[1],
                            "default": column[2] if len(column) > 2 else None,
                            "compression_codec": column[3] if len(column) > 3 else None,
                            "ttl": column[4] if len(column) > 4 else None,
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
            client, lock = self._get_connection()
            if not client:
                return "No available database connection"

            with lock:
                # Get list of tables
                tables = client.execute("SHOW TABLES")
                return [table[0] for table in tables]

        except Exception as e:
            error_msg = f"Error getting available tables: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def flush_all_queues(self):
        """Force flush all pending items in queues"""
        # Process remaining trades
        batch = []
        while not self.trades_queue.empty():
            try:
                batch.append(self.trades_queue.get_nowait())
            except:
                break
        if batch:
            self._process_batch_immediate(batch, "trades")

        # Process remaining iterations
        batch = []
        while not self.iterations_queue.empty():
            try:
                batch.append(self.iterations_queue.get_nowait())
            except:
                break
        if batch:
            self._process_batch_immediate(batch, "iterations")

        # Process remaining sub-iterations
        batch = []
        while not self.sub_iterations_queue.empty():
            try:
                batch.append(self.sub_iterations_queue.get_nowait())
            except:
                break
        if batch:
            self._process_batch_immediate(batch, "sub_iterations")

    def _process_batch_immediate(self, batch: List[Dict[str, Any]], table: str):
        """Process a batch immediately without waiting"""
        if not batch:
            return

        validated_batch = self._validate_batch_data(batch, table)
        if validated_batch:
            client, lock = self._get_connection()
            if client:
                try:
                    with lock:
                        client.execute(f"INSERT INTO {table} VALUES", validated_batch)
                except Exception as e:
                    logger.error(
                        f"Error processing immediate batch for {table}: {str(e)}"
                    )

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.running = False

        # Flush remaining data
        try:
            self.flush_all_queues()
        except:
            pass

        # Wait for threads to complete
        if hasattr(self, "trades_thread"):
            self.trades_thread.join(timeout=2)
        if hasattr(self, "iterations_thread"):
            self.iterations_thread.join(timeout=2)
        if hasattr(self, "sub_iterations_thread"):
            self.sub_iterations_thread.join(timeout=2)

        # Close all connections
        for i, client in enumerate(self.connection_pool):
            try:
                with self.connection_locks[i]:
                    client.disconnect()
            except:
                pass
