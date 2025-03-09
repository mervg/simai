"""
State management module for WebSIMAI application.

This module contains functions for initializing and managing the
Streamlit session state, including filter application and state updates.
"""
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
from loguru import logger

from logger.logger import LoggingContext


def initialize_state() -> None:
    """
    Initialize the Streamlit session state.
    
    This function sets up all necessary session state variables
    if they don't already exist.
    """
    with LoggingContext("S0130 - Session State Initialization"):
        try:
            # Initialize app status
            if 'app_status' not in st.session_state:
                st.session_state.app_status = {
                    'data_loaded': False,
                    'error': None,
                }
            
            # Initialize filters
            if 'filters' not in st.session_state:
                st.session_state.filters = {}
            
            # Initialize filter options
            if 'filter_options' not in st.session_state:
                st.session_state.filter_options = {}
            
            # Initialize filtered DataFrame info
            if 'filtered_df_info' not in st.session_state:
                st.session_state.filtered_df_info = {
                    'count': 0,
                    'page': 0,
                    'page_size': 10,
                }
            
            # Initialize selected record
            if 'selected_record' not in st.session_state:
                st.session_state.selected_record = None
            
            # Initialize status message
            if 'status' not in st.session_state:
                st.session_state.status = "Ready"
            
            logger.key("S0130 - OK: Session state initialized successfully")
            
        except Exception as e:
            logger.error(f"S0130 - FAILED: Error initializing session state: {str(e)}")


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to the DataFrame.
    
    Args:
        df: Original DataFrame
        filters: Dictionary of column-value filters
        
    Returns:
        Filtered DataFrame
    """
    with LoggingContext("S0310 - Filter Application"):
        try:
            # Start with a copy of the original DataFrame
            filtered_df = df.copy()
            
            # Skip if no filters
            if not filters:
                logger.info("No filters applied")
                logger.key("S0310 - OK: No filters applied")
                return filtered_df
            
            # Apply each filter
            for column, value in filters.items():
                if value and column in filtered_df.columns:
                    # Handle list of values (multi-select)
                    if isinstance(value, list):
                        if value:  # Only apply if list is not empty
                            filtered_df = filtered_df[filtered_df[column].isin(value)]
                            logger.info(f"Applied filter: {column} in {value}")
                    # Handle single value
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
                        logger.info(f"Applied filter: {column} = {value}")
            
            logger.info(f"Filtered DataFrame: {len(filtered_df)} records (from {len(df)})")
            logger.key("S0310 - OK: Filters applied successfully")
            return filtered_df
            
        except Exception as e:
            logger.error(f"S0310 - FAILED: Error applying filters: {str(e)}")
            # Return original DataFrame on error
            return df


def update_filter(column: str, value: Any) -> None:
    """
    Update a filter in the session state.
    
    Args:
        column: Column name to filter on
        value: Value to filter by
    """
    with LoggingContext("S0320 - Filter Update"):
        try:
            # Update filter in session state
            st.session_state.filters[column] = value
            
            # Reset pagination when filter changes
            st.session_state.filtered_df_info['page'] = 0
            
            logger.info(f"Updated filter: {column} = {value}")
            logger.key("S0320 - OK: Filter updated successfully")
            
        except Exception as e:
            logger.error(f"S0320 - FAILED: Error updating filter: {str(e)}")


def clear_filters() -> None:
    """
    Clear all filters from the session state.
    """
    with LoggingContext("S0320 - Filter Clear"):
        try:
            # Clear filters
            st.session_state.filters = {}
            
            # Reset pagination
            st.session_state.filtered_df_info['page'] = 0
            
            logger.info("Cleared all filters")
            logger.key("S0320 - OK: Filters cleared successfully")
            
        except Exception as e:
            logger.error(f"S0320 - FAILED: Error clearing filters: {str(e)}")


def update_pagination(page: int, page_size: int = None) -> None:
    """
    Update pagination settings in the session state.
    
    Args:
        page: Page number to navigate to
        page_size: Number of records per page (optional)
    """
    with LoggingContext("S0340 - Pagination Update"):
        try:
            # Update page number
            st.session_state.filtered_df_info['page'] = max(0, page)
            
            # Update page size if provided
            if page_size is not None:
                st.session_state.filtered_df_info['page_size'] = max(1, page_size)
            
            logger.info(f"Updated pagination: page={page}, page_size={st.session_state.filtered_df_info['page_size']}")
            logger.key("S0340 - OK: Pagination updated successfully")
            
        except Exception as e:
            logger.error(f"S0340 - FAILED: Error updating pagination: {str(e)}")


def select_record(record_id: Any) -> None:
    """
    Select a record for detailed view.
    
    Args:
        record_id: ID of the record to select
    """
    with LoggingContext("S0350 - Record Selection"):
        try:
            # Update selected record in session state
            st.session_state.selected_record = record_id
            
            logger.info(f"Selected record: {record_id}")
            logger.key("S0350 - OK: Record selected successfully")
            
        except Exception as e:
            logger.error(f"S0350 - FAILED: Error selecting record: {str(e)}")
