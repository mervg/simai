"""
Data access module for WebSIMAI application.

This module contains functions for connecting to the SQLite database,
loading UDF data into a pandas DataFrame, and preparing filter options.
"""
import os
import sqlite3
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import streamlit as st
from loguru import logger

from logger.logger import LoggingContext


def get_sqlite_connection(db_path: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Establish a connection to the SQLite database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Tuple containing the connection and cursor objects
        
    Raises:
        sqlite3.Error: If connection to the database fails
    """
    with LoggingContext("S0210 - SQLite Connection"):
        try:
            # Check if database file exists
            if not os.path.exists(db_path):
                logger.error(f"S0210 - FAILED: Database file not found at {db_path}")
                raise FileNotFoundError(f"Database file not found at {db_path}")
            
            # Establish connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test connection
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()
            logger.info(f"SQLite version: {version[0]}")
            
            logger.key("S0210 - OK: SQLite connection established successfully")
            return conn, cursor
            
        except sqlite3.Error as e:
            logger.error(f"S0210 - FAILED: SQLite connection error: {str(e)}")
            raise sqlite3.Error(f"Failed to connect to database: {str(e)}")


@st.cache_data(ttl=3600)
def load_udf_dataframe(cursor: sqlite3.Cursor) -> pd.DataFrame:
    """
    Load UDF data from SQLite into a pandas DataFrame.
    
    This function is cached by Streamlit to improve performance.
    
    Args:
        cursor: SQLite cursor object
        
    Returns:
        DataFrame containing UDF data
        
    Raises:
        ValueError: If data loading fails
    """
    with LoggingContext("S0220 - UDF Data Loading"):
        try:
            # Execute query to get all UDF data
            query = """
            SELECT * FROM UDF
            """
            
            # Load data into DataFrame
            df = pd.read_sql_query(query, cursor.connection)
            
            # Log data loading statistics
            logger.info(f"Loaded {len(df)} UDF records")
            logger.info(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            
            logger.key("S0220 - OK: UDF data loaded successfully")
            return df
            
        except Exception as e:
            logger.error(f"S0220 - FAILED: Error loading UDF data: {str(e)}")
            raise ValueError(f"Failed to load UDF data: {str(e)}")


def prepare_filter_options(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """
    Prepare filter options from the DataFrame.
    
    Extracts unique values for each filterable column.
    
    Args:
        df: DataFrame containing UDF data
        
    Returns:
        Dictionary mapping column names to lists of unique values
        
    Raises:
        ValueError: If filter preparation fails
    """
    with LoggingContext("S0240 - Filter Options Preparation"):
        try:
            filter_options = {}
            
            # Define filterable columns
            filterable_columns = [
                'UDF_SCHEMA', 'UDF_NAME', 'UDF_LANGUAGE', 'UDF_TYPE',
                'UDF_OWNER', 'UDF_CREATED_BY'
            ]
            
            # Extract unique values for each column
            for column in filterable_columns:
                if column in df.columns:
                    # Get unique values and sort them
                    unique_values = sorted(df[column].dropna().unique().tolist())
                    filter_options[column] = unique_values
                    logger.info(f"Prepared {len(unique_values)} filter options for {column}")
            
            logger.key("S0240 - OK: Filter options prepared successfully")
            return filter_options
            
        except Exception as e:
            logger.error(f"S0240 - FAILED: Error preparing filter options: {str(e)}")
            raise ValueError(f"Failed to prepare filter options: {str(e)}")
