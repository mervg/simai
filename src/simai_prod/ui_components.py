"""
UI components module for WebSIMAI application.

This module contains functions for creating the main layout,
sidebar, data explorer tab, export tab, and logs tab.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
from loguru import logger

from logger.logger import LoggingContext
from simai_prod.state_management import update_filter, clear_filters, update_pagination, select_record
from simai_prod.export import df_to_structured_json, generate_json_file


def create_main_layout() -> None:
    """
    Create the main application layout.
    
    This function sets up the page configuration and main container.
    """
    with LoggingContext("S0410 - Main Layout Creation"):
        try:
            # Set page configuration
            st.set_page_config(
                page_title="WebSIMAI - UDF Explorer",
                page_icon="📊",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Create main title
            st.title("WebSIMAI - UDF Explorer")
            
            # Create tabs
            tab_names = ["Data Explorer", "Export", "Logs"]
            tabs = st.tabs(tab_names)
            
            # Store tabs in session state for reference
            st.session_state.tabs = tabs
            
            # Display status
            st.sidebar.text(f"Status: {st.session_state.status}")
            
            logger.key("S0410 - OK: Main layout created successfully")
            
        except Exception as e:
            logger.error(f"S0410 - FAILED: Error creating main layout: {str(e)}")


def create_sidebar(df: pd.DataFrame, filter_options: Dict[str, List[Any]]) -> None:
    """
    Create the sidebar with filter controls.
    
    Args:
        df: DataFrame containing UDF data
        filter_options: Dictionary of filter options
    """
    with LoggingContext("S0420 - Sidebar Creation"):
        try:
            # Add sidebar header
            st.sidebar.header("Filters")
            
            # Add record count information
            total_records = len(df)
            filtered_records = st.session_state.filtered_df_info['count']
            st.sidebar.text(f"Showing {filtered_records} of {total_records} records")
            
            # Add filter controls
            for column, options in filter_options.items():
                # Create a more user-friendly label
                label = column.replace('_', ' ').title()
                
                # Create a multi-select widget
                selected_values = st.sidebar.multiselect(
                    label=label,
                    options=options,
                    default=st.session_state.filters.get(column, [])
                )
                
                # Update filter in session state when changed
                if selected_values != st.session_state.filters.get(column, []):
                    update_filter(column, selected_values)
            
            # Add clear filters button
            if st.sidebar.button("Clear Filters"):
                clear_filters()
            
            logger.key("S0420 - OK: Sidebar created successfully")
            
        except Exception as e:
            logger.error(f"S0420 - FAILED: Error creating sidebar: {str(e)}")


def create_data_explorer_tab(df: pd.DataFrame) -> None:
    """
    Create the data explorer tab.
    
    Args:
        df: Filtered DataFrame to display
    """
    with LoggingContext("S0430 - Data Explorer Tab Creation"):
        try:
            # Get the data explorer tab
            tab = st.session_state.tabs[0]
            
            # Use the tab context
            with tab:
                # Handle pagination
                page = st.session_state.filtered_df_info['page']
                page_size = st.session_state.filtered_df_info['page_size']
                
                # Calculate total pages
                total_pages = (len(df) - 1) // page_size + 1 if len(df) > 0 else 1
                
                # Create pagination controls
                col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
                
                with col1:
                    if st.button("⏮️", disabled=page <= 0):
                        update_pagination(0)
                
                with col2:
                    if st.button("⏪ Previous", disabled=page <= 0):
                        update_pagination(page - 1)
                
                with col3:
                    if st.button("Next ⏩", disabled=page >= total_pages - 1):
                        update_pagination(page + 1)
                
                with col4:
                    if st.button("⏭️", disabled=page >= total_pages - 1):
                        update_pagination(total_pages - 1)
                
                # Display page information
                st.text(f"Page {page + 1} of {total_pages}")
                
                # Display records for current page
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(df))
                
                if len(df) > 0:
                    # Get records for current page
                    page_df = df.iloc[start_idx:end_idx].copy()
                    
                    # Display records
                    st.dataframe(
                        page_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Create record detail view
                    st.subheader("Record Details")
                    
                    # Create record selector
                    selected_index = st.selectbox(
                        "Select a record to view details",
                        options=range(len(page_df)),
                        format_func=lambda i: f"Record {start_idx + i + 1}"
                    )
                    
                    # Display selected record
                    if selected_index is not None:
                        record = page_df.iloc[selected_index]
                        select_record(record.name)
                        
                        # Display record details
                        st.json(record.to_dict())
                else:
                    st.info("No records match the current filters.")
            
            logger.key("S0430 - OK: Data explorer tab created successfully")
            
        except Exception as e:
            logger.error(f"S0430 - FAILED: Error creating data explorer tab: {str(e)}")


def create_export_tab(df: pd.DataFrame) -> None:
    """
    Create the export tab.
    
    Args:
        df: Filtered DataFrame to export
    """
    with LoggingContext("S0440 - Export Tab Creation"):
        try:
            # Get the export tab
            tab = st.session_state.tabs[1]
            
            # Use the tab context
            with tab:
                st.header("Export Data")
                
                # Display record count
                st.info(f"Exporting {len(df)} records")
                
                # Create export options
                export_format = st.selectbox(
                    "Export Format",
                    options=["JSON"],
                    index=0
                )
                
                # Create filename input
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_filename = f"udf_export_{timestamp}.json"
                
                filename = st.text_input(
                    "Filename",
                    value=default_filename
                )
                
                # Create export button
                if st.button("Generate Export"):
                    with st.spinner("Generating export..."):
                        try:
                            # Convert DataFrame to JSON
                            json_data = df_to_structured_json(df)
                            
                            # Generate JSON file
                            file_path = generate_json_file(json_data, filename)
                            
                            # Provide download link
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="Download Export",
                                    data=f,
                                    file_name=filename,
                                    mime="application/json"
                                )
                            
                            st.success(f"Export generated successfully: {filename}")
                            
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                            logger.error(f"Export failed: {str(e)}")
            
            logger.key("S0440 - OK: Export tab created successfully")
            
        except Exception as e:
            logger.error(f"S0440 - FAILED: Error creating export tab: {str(e)}")


def create_logs_tab() -> None:
    """
    Create the logs tab.
    """
    with LoggingContext("S0450 - Logs Tab Creation"):
        try:
            # Get the logs tab
            tab = st.session_state.tabs[2]
            
            # Use the tab context
            with tab:
                st.header("Application Logs")
                
                # Create log level filter
                log_level = st.selectbox(
                    "Log Level",
                    options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    index=1  # Default to INFO
                )
                
                # Get log buffer from session state
                if 'log_buffer' in st.session_state:
                    logs = st.session_state.log_buffer
                    
                    # Filter logs by level
                    level_map = {
                        "DEBUG": 0,
                        "INFO": 1,
                        "WARNING": 2,
                        "ERROR": 3,
                        "CRITICAL": 4
                    }
                    
                    selected_level = level_map.get(log_level, 1)
                    filtered_logs = [
                        log for log in logs 
                        if level_map.get(log['level'], 0) >= selected_level
                    ]
                    
                    # Display logs
                    if filtered_logs:
                        for log in filtered_logs:
                            level = log['level']
                            time = log['time']
                            message = log['message']
                            context = log['context']
                            
                            # Format log entry based on level
                            if level == "ERROR" or level == "CRITICAL":
                                st.error(f"{time} | {level} | {context} | {message}")
                            elif level == "WARNING":
                                st.warning(f"{time} | {level} | {context} | {message}")
                            else:
                                st.info(f"{time} | {level} | {context} | {message}")
                    else:
                        st.info(f"No logs with level {log_level} or higher.")
                else:
                    st.info("No logs available.")
            
            logger.key("S0450 - OK: Logs tab created successfully")
            
        except Exception as e:
            logger.error(f"S0450 - FAILED: Error creating logs tab: {str(e)}")
