"""
Logging integration module for WebSIMAI application.

This module contains classes and functions for integrating
the existing logging system with Streamlit UI.
"""
import streamlit as st
from loguru import logger

from logger.logger import LoggingContext


class StreamlitLogHandler:
    """
    Custom log handler that captures logs for display in Streamlit.
    
    This handler maintains a buffer of log records in the Streamlit
    session state for display in the UI.
    """
    
    def __init__(self, level="INFO", buffer_size=100):
        """
        Initialize the handler.
        
        Args:
            level: Minimum log level to capture
            buffer_size: Maximum number of log records to keep in buffer
        """
        self.level = level
        self.buffer_size = buffer_size
        
        # Initialize the log buffer in session state if it doesn't exist
        if 'log_buffer' not in st.session_state:
            st.session_state.log_buffer = []
    
    def write(self, record):
        """
        Write a log record to the buffer.
        
        This method is called by the loguru handler.
        
        Args:
            record: Log record dictionary
        """
        # Extract relevant information from the record
        log_entry = {
            'level': record["level"].name,
            'time': record["time"].strftime("%Y-%m-%d %H:%M:%S"),
            'message': record["message"],
            'context': record.get("extra", {}).get("context", "")
        }
        
        # Add to the buffer
        st.session_state.log_buffer.append(log_entry)
        
        # Trim buffer if it exceeds buffer_size
        while len(st.session_state.log_buffer) > self.buffer_size:
            st.session_state.log_buffer.pop(0)


def configure_streamlit_logging():
    """
    Configure loguru to send logs to the Streamlit UI.
    
    This function adds a custom handler to the loguru logger
    that captures logs for display in the Streamlit UI.
    """
    with LoggingContext("S0120 - Streamlit Logging Configuration"):
        try:
            # Create a custom handler
            handler = StreamlitLogHandler(level="INFO", buffer_size=100)
            
            # Add the handler to loguru
            logger.add(
                handler.write,
                level="INFO",
                format="{time} | {level} | {message} | {extra}",
                serialize=True  # Pass the record as a dict to the handler
            )
            
            logger.key("S0120 - OK: Streamlit logging configured successfully")
            
        except Exception as e:
            logger.error(f"S0120 - FAILED: Error configuring Streamlit logging: {str(e)}")


class StreamlitLoggingContext(LoggingContext):
    """
    Extended LoggingContext for Streamlit UI updates.
    
    This context manager not only switches the logging context
    but also updates the UI to reflect the current operation.
    """
    
    def __init__(self, context_name: str, ui_update: bool = True):
        """
        Initialize the context.
        
        Args:
            context_name: Name of the logging context
            ui_update: Whether to update the UI status
        """
        super().__init__(context_name)
        self.ui_update = ui_update
        self.previous_status = None
    
    def __enter__(self):
        """Enter the context, updating both logging context and UI status."""
        # Call parent's enter method to update logging context
        result = super().__enter__()
        
        # Update UI status if requested
        if self.ui_update and hasattr(st.session_state, 'status'):
            # Store previous status
            self.previous_status = st.session_state.get('status')
            
            # Update status to current context
            st.session_state.status = self.context_name
        
        return result
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, restoring both logging context and UI status."""
        # Call parent's exit method to restore logging context
        result = super().__exit__(exc_type, exc_val, exc_tb)
        
        # Restore UI status if requested
        if self.ui_update and hasattr(st.session_state, 'status') and self.previous_status:
            st.session_state.status = self.previous_status
        
        return result
