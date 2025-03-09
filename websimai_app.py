"""
WebSIMAI Application.

This is the main entry point for the WebSIMAI application.
It orchestrates the entire application flow, including initialization,
data loading, UI rendering, and event handling.
"""
import os
from datetime import datetime
import streamlit as st
from loguru import logger

# Import logger initialization
from logger.logger import LoggingContext, initialize_logging

# Import modules
from simai_prod.data_access import get_sqlite_connection, load_udf_dataframe, prepare_filter_options
from simai_prod.state_management import initialize_state, apply_filters
from simai_prod.ui_components import create_main_layout, create_sidebar, create_data_explorer_tab, create_export_tab, create_logs_tab
from simai_prod.logging_integration import configure_streamlit_logging, StreamlitLoggingContext


def main():
    """
    Main application entry point.
    
    This function orchestrates the entire application flow:
    1. Initialization
    2. Data loading
    3. UI rendering
    4. Event handling
    """
    try:
        # S0100 - Initialization and Configuration
        with LoggingContext("S0100 - Initialization"):
            # S0110 - Load environment variables and configuration
            with LoggingContext("S0110 - Configuration Loading"):
                # Set up database path
                SQLITE_DB_PATH = os.path.dirname(__file__)
                SQLITE_DB_FILENAME = os.path.join(SQLITE_DB_PATH, 'UDF.db')
                
                # Store in session state for reference
                st.session_state.db_path = SQLITE_DB_FILENAME
                
                logger.key("S0110 - OK: Configuration loaded successfully")
            
            # S0120 - Initialize logging system
            with LoggingContext("S0120 - Logging Initialization"):
                # Initialize logging
                initialize_logging('logger/logging_config.yaml')
                
                # Configure Streamlit logging
                configure_streamlit_logging()
                
                logger.key("S0120 - OK: Logging initialized successfully")
            
            # Initialize session state
            initialize_state()
            
            logger.key("S0100 - OK: Initialization complete")
        
        # Create main layout first (must be called before any other UI elements)
        create_main_layout()
        
        # S0200 - Data Loading and Preparation
        with LoggingContext("S0200 - Data Loading"):
            # Check if we've already loaded data (Streamlit rerun protection)
            df_udf = None
            filter_options = None
            
            if not st.session_state.app_status.get('data_loaded', False):
                # S0210 - Open SQLite connection
                with LoggingContext("S0210 - SQLite Connection"):
                    try:
                        # Get database connection
                        conn, cursor = get_sqlite_connection(st.session_state.db_path)
                        logger.key("S0210 - OK: SQLite connection established")
                        
                        # S0220 - Load UDF data into DataFrame
                        with LoggingContext("S0220 - UDF Data Loading"):
                            # Load data with caching
                            df_udf = load_udf_dataframe(cursor)
                            
                            # Store in session state
                            st.session_state.df_udf = df_udf
                            
                            # Update application status
                            st.session_state.app_status['data_loaded'] = True
                            
                            logger.key("S0220 - OK: UDF data loaded successfully")
                        
                        # S0230 - Close SQLite connection
                        with LoggingContext("S0230 - SQLite Cleanup"):
                            # Close database connection
                            conn.close()
                            logger.key("S0230 - OK: SQLite connection closed")
                        
                        # S0240 - Prepare filter options
                        with LoggingContext("S0240 - Filter Options Preparation"):
                            # Extract filter options
                            filter_options = prepare_filter_options(df_udf)
                            
                            # Store in session state
                            st.session_state.filter_options = filter_options
                            
                            logger.key("S0240 - OK: Filter options prepared successfully")
                    
                    except Exception as e:
                        logger.critical(f"S0200 - FAILED: Data loading error: {str(e)}")
                        st.error(f"Failed to load data: {str(e)}")
                        
                        # Update application status
                        st.session_state.app_status['data_loaded'] = False
                        st.session_state.app_status['error'] = str(e)
            else:
                # Retrieve data from session state
                df_udf = st.session_state.df_udf
                filter_options = st.session_state.filter_options
            
            logger.key("S0200 - OK: Data loading and preparation complete")
        
        # S0300 - Data Processing
        with LoggingContext("S0300 - Data Processing"):
            # Apply filters to the DataFrame
            filtered_df = apply_filters(df_udf, st.session_state.filters)
            
            # Update filtered count in session state
            st.session_state.filtered_df_info['count'] = len(filtered_df)
            
            logger.key("S0300 - OK: Data processing complete")
        
        # S0400 - UI Components
        with LoggingContext("S0400 - UI Rendering"):
            # Create sidebar
            create_sidebar(df_udf, filter_options)
            
            # Create data explorer tab
            create_data_explorer_tab(filtered_df)
            
            # Create export tab
            create_export_tab(filtered_df)
            
            # Create logs tab
            create_logs_tab()
            
            logger.key("S0400 - OK: UI rendering complete")
    
    except Exception as e:
        logger.critical(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
