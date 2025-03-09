"""Advanced Logging System for AI Task Analyzer.

This module provides a sophisticated, context-aware logging system using Loguru to replace
scattered print statements and enhance code traceability.

The system supports:
- Multiple output streams (console, run file, context-specific files, visualization, errors)
- Custom log levels (DEBUG, INFO, KEY, WARNING, ERROR, CRITICAL)
- Context-aware logging with automatic file routing
- Plain text visualization output
- Hierarchical configuration via YAML

Typical usage:
    from src.logger import logger, LoggingContext, VizContext, initialize_logging
    
    # Initialize with default or custom config
    initialize_logging()
    
    # Basic logging
    logger.info("Regular log message")
    
    # Context-specific logging
    with LoggingContext("process_name"):
        logger.info("This goes to both console and process_name.log")
        
    # Visualization output
    with VizContext("chart_output"):
        logger.info("Plain text visualization data")
"""

import os
import re
import yaml
import contextvars
from typing import Any, Optional
import sys
from loguru import logger
import datetime

# Constants
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logging_config.yaml")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Global variables
context_stack = contextvars.ContextVar("logging_context_stack", default=[])
context_sinks = {}  # Mapping of context names to sink IDs
global_config = None  # Global configuration instance

# Function to add context to extra record data for formatting
def add_context_to_record(record):
    """Add the current context name to the log record's extra field.
    
    This allows using {extra[context]} in log format strings.
    
    Args:
        record: The log record to modify
    
    Returns:
        Modified record with context information
    """
    # Get current context
    context = get_current_context()
    
    # Add context to record's extra field
    record["extra"]["context"] = context or "main"
    
    return record

# Step 1: Define custom log levels

# Define KEY log level (between INFO and WARNING)
KEY_LEVEL_NAME = "KEY"
KEY_LEVEL_NO = 25  # Between INFO (20) and WARNING (30)
logger.level(KEY_LEVEL_NAME, no=KEY_LEVEL_NO, color="<magenta>")

# Add KEY method to logger (using Loguru's opt functionality to preserve caller info)
def key(self, message, *args, **kwargs):
    """Log key information (level between INFO and WARNING).
    
    Args:
        message: The log message
        *args: Arguments to be formatted in the message
        **kwargs: Keyword arguments to be formatted in the message
    """
    # Use opt(depth=1) to capture the caller's frame instead of this function
    self.opt(depth=1).log(KEY_LEVEL_NAME, message, *args, **kwargs)

# Add the key method to Logger class
logger.__class__.key = key

# Add with_context method to temporarily switch context for a single log message
def with_context(self, temp_context):
    """Create a logger proxy that temporarily uses a different context for the next log message.
    
    Args:
        temp_context: The temporary context name to use for the next log message
        
    Returns:
        A logger proxy object with the temporary context
    """
    class LoggerContextProxy:
        def __init__(self, logger_instance, context_name):
            self.logger = logger_instance
            self.context_name = context_name
            
        def _log_with_temp_context(self, level, message, *args, **kwargs):
            # Save the current context stack
            original_stack = context_stack.get()
            
            try:
                # Set temporary context
                if self.context_name == "main":
                    # Empty stack for main context
                    context_stack.set([])
                else:
                    # Use the specified context
                    context_stack.set([self.context_name])
                
                # Log the message with the temporary context
                # Use opt(depth=2) to capture the caller's frame (not this proxy method)
                self.logger.opt(depth=2).log(level, message, *args, **kwargs)
            finally:
                # Restore the original context stack
                context_stack.set(original_stack)
        
        # Implement all log level methods
        def debug(self, message, *args, **kwargs):
            self._log_with_temp_context("DEBUG", message, *args, **kwargs)
            
        def info(self, message, *args, **kwargs):
            self._log_with_temp_context("INFO", message, *args, **kwargs)
            
        def key(self, message, *args, **kwargs):
            self._log_with_temp_context(KEY_LEVEL_NAME, message, *args, **kwargs)
            
        def warning(self, message, *args, **kwargs):
            self._log_with_temp_context("WARNING", message, *args, **kwargs)
            
        def error(self, message, *args, **kwargs):
            self._log_with_temp_context("ERROR", message, *args, **kwargs)
            
        def critical(self, message, *args, **kwargs):
            self._log_with_temp_context("CRITICAL", message, *args, **kwargs)
    
    return LoggerContextProxy(self, temp_context)

# Add the with_context method to Logger class
logger.__class__.with_context = with_context

# Step 2: Implement Configuration Management Class

class LoggingConfig:
    """Configuration manager for the logging system.
    
    Handles loading and accessing configuration from YAML files with hierarchical overrides.
    Configuration hierarchy (highest to lowest priority):
    1. Context-specific settings
    2. Stream-specific settings
    3. General defaults
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize the logging configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = {}
        self.config_path = config_path
        # Store the directory of the config file for resolving relative paths
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file.
        
        Falls back to default configuration if file not found or has errors.
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Configuration file not found: {self.config_path}, using defaults")
                self._create_default_config()
                return
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Process paths in the configuration
            self._process_path_config()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self._create_default_config()
            
    def _process_path_config(self) -> None:
        """Process path configurations to handle relative paths.
        
        Validates paths but preserves relative paths in the configuration.
        """
        # Process log directory paths
        for path_key in ["general_defaults.log_dir_run", "general_defaults.log_dir_viz"]:
            path_value = self.get_value(path_key)
            if path_value:
                # Normalize the path but keep it relative if it was relative
                norm_path = os.path.normpath(path_value)
                
                # Update the configuration with the normalized path
                parts = path_key.split('.')
                if len(parts) == 2:
                    section, key = parts
                    if section in self.config and key in self.config[section]:
                        self.config[section][key] = norm_path

    def _create_default_config(self) -> None:
        """Create default configuration when no file is available."""
        # Basic default configuration
        self.config = {
            "general_defaults": {
                "level": "INFO",
                "log_msg_format": DEFAULT_FORMAT,
                "log_dir_run": "logs/run",
                "log_dir_viz": "logs/viz",
                "console_enabled": True,
                "file_enabled": True,
                "error_file_enabled": True,
                "run_file_enabled": True,
                "process_context_default_filename": "%[context_name]",
                "viz_context_default_filename": "%[viz_name].txt"
            },
            "streams": {
                "console": {
                    "level": "INFO"
                },
                "run_file": {
                    "level": "DEBUG"
                },
                "error_file": {
                    "level": "WARNING"
                }
            },
            "logging_contexts": {}
        }
            
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Get a value from the configuration using a dot-notation path.
        
        Args:
            key_path: Dot-notation path to the configuration value
            default: Default value if the key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_context_value(self, context_name: str, key: str, default: Any = None) -> Any:
        """Get a context-specific configuration value with hierarchy fallback.
        
        Looks for the value in:
        1. Context-specific settings
        2. Stream-specific settings
        3. General defaults
        
        Args:
            context_name: Name of the logging context
            key: Configuration key to look for
            default: Default value if the key is not found
            
        Returns:
            The configuration value or default
        """
        # Try context-specific setting
        try:
            return self.config["logging_contexts"][context_name][key]
        except (KeyError, TypeError):
            pass
            
        # Try stream-specific setting (defaults to "context_file" stream)
        try:
            return self.config["streams"]["context_file"][key]
        except (KeyError, TypeError):
            pass
            
        # Fall back to general defaults
        try:
            return self.config["general_defaults"][key]
        except (KeyError, TypeError):
            return default
            
    def get_stream_value(self, stream_name: str, key: str, default: Any = None) -> Any:
        """Get a stream-specific configuration value with fallback to defaults.
        
        Args:
            stream_name: Name of the stream
            key: Configuration key to look for
            default: Default value if the key is not found
            
        Returns:
            The configuration value or default
        """
        # Try stream-specific setting
        try:
            return self.config["streams"][stream_name][key]
        except (KeyError, TypeError):
            pass
            
        # Fall back to general defaults
        try:
            return self.config["general_defaults"][key]
        except (KeyError, TypeError):
            return default

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the config file directory if it's relative.
        
        Args:
            path: The path to resolve
            
        Returns:
            Absolute path if input was relative, otherwise unchanged path
        """
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(self.config_dir, path))
        return path

# Helper functions for log formatting and processing

def strip_ansi_escape_sequences(text: str) -> str:
    """Remove ANSI escape sequences from text.
    
    Used to strip color codes from logs when writing to files.
    
    Args:
        text: Text containing ANSI escape sequences
        
    Returns:
        Text with ANSI escape sequences removed
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Step 3: Implement Context Management

# Step 3.1: Base Context Class
class BaseLoggingContext:
    """Base class for logging context managers.
    
    Provides common functionality for context management.
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """Initialize the base logging context.
        
        Args:
            config: Optional LoggingConfig instance
        """
        # Use provided config, global config, or create a new one
        self.config = config or global_config or LoggingConfig()
        self.sink_ids = []
        self.sink_creation_failed = False
        
    def __enter__(self):
        """Enter the context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up sinks."""
        # Remove all sinks created by this context
        for sink_id in self.sink_ids:
            try:
                logger.remove(sink_id)
            except Exception:
                pass  # Sink may already be removed

# Function to get current context name from stack
def get_current_context() -> Optional[str]:
    """Get the name of the current logging context.
    
    Returns:
        Current context name or None if not in a context
    """
    stack = context_stack.get()
    return stack[-1] if stack else None
    
# Add context prefix to messages
def with_context_prefix(message: str) -> str:
    """Add context prefix to a log message.
    
    Args:
        message: Original log message
        
    Returns:
        Message with context prefix
    """
    context = get_current_context()
    if context:
        return f"[{context}] {message}"
    return message

# Step 3.2: Process Logging Context

class LoggingContext(BaseLoggingContext):
    """Context manager for process-specific logging.
    
    Creates a separate log file for each context and adds context prefix to messages.
    """
    
    def __init__(self, context_name: str, config: Optional[LoggingConfig] = None):
        """Initialize the logging context.
        
        Args:
            context_name: Name of the context
            config: Optional LoggingConfig instance
        """
        super().__init__(config)
        self.context_name = context_name
        
    def __enter__(self):
        """Enter the context.
        
        Pushes the context name onto the stack and creates a context-specific sink.
        """
        # Get current context stack and add this context
        stack = context_stack.get()
        new_stack = stack + [self.context_name]
        context_stack.set(new_stack)
        
        # Create a sink for this context if it doesn't exist
        if self.context_name not in context_sinks:
            self._create_context_sink()
        else:
            # Add existing sink to this instance for proper cleanup
            self.sink_ids.append(context_sinks[self.context_name])
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context.
        
        Pops the context name from the stack but preserves the sink for future use.
        """
        # Get current context stack and remove this context
        stack = context_stack.get()
        new_stack = stack[:-1]  # Remove the last element
        context_stack.set(new_stack)
        
        # We don't remove or disable the sinks when exiting
        # This allows the same context to be reused later
        # The filtering of messages is handled by the context_stack mechanism
        
    def _create_context_sink(self) -> None:
        """Create a file sink for this context.
        
        Creates a log file based on the context name and configuration.
        """
        try:
            # Get configuration for this context
            log_level = self.config.get_context_value(self.context_name, "level", DEFAULT_LOG_LEVEL)
            log_format = self.config.get_context_value(self.context_name, "log_msg_format", 
                                                     self.config.get_value("streams.context_file.log_msg_format", DEFAULT_FORMAT))
            
            # Process context filename template
            filename_template = self.config.get_value("general_defaults.process_context_default_filename",
                                                     "%[context_name]")
            filename = filename_template.replace("%[context_name]", self.context_name)
            
            # Ensure .log extension
            if not filename.endswith(".log"):
                filename += ".log"
                
            # Get log directory from global config (which has the timestamped path)
            log_dir = global_config.get_value("general_defaults.log_dir_run")
            # Normalize path separators for the OS
            log_dir = os.path.normpath(log_dir)
            filepath = os.path.join(log_dir, filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Define a filter function that only allows messages when this context is active
            context_name = self.context_name  # Capture for closure
            def context_filter(record):
                current_context = get_current_context()
                return current_context == context_name
            
            # Create sink with formatting from config and context filter
            sink_id = logger.add(
                sink=filepath,
                level=log_level,
                format=strip_ansi_escape_sequences(log_format),
                filter=context_filter,  # Only log when this context is active
                enqueue=True,  # Use queue for thread safety
                backtrace=True,  # Include traceback in errors
                diagnose=True,  # Include variables in traceback
            )
            
            # Track the sink
            self.sink_ids.append(sink_id)
            
            # Register this sink in the global registry
            context_sinks[self.context_name] = sink_id
            
        except Exception as e:
            logger.error(f"Failed to create context sink for {self.context_name}: {e}")
            self.sink_creation_failed = True

# Step 3.3: Visualization Context

class VizContext(BaseLoggingContext):
    """Context manager for logging visualization output.
    
    Creates a plain text sink without log formatting for visualization output.
    """
    
    def __init__(self, viz_name: str, config: Optional[LoggingConfig] = None):
        """Initialize the visualization context.
        
        Args:
            viz_name: Name of the visualization
            config: Optional LoggingConfig instance
        """
        super().__init__(config)
        self.viz_name = viz_name
        self.context_name = f"viz_{viz_name}"  # Context name for config lookup
        
    def __enter__(self):
        """Enter the context.
        
        Pushes the context name onto the stack and creates a visualization-specific sink.
        """
        # Get current context stack and add this context
        stack = context_stack.get()
        new_stack = stack + [self.context_name]
        context_stack.set(new_stack)
        
        # Create a sink for this visualization
        self._create_viz_sink()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up sinks.
        
        Pops the context name from the stack and removes the sink.
        """
        # Get current context stack and remove this context
        stack = context_stack.get()
        new_stack = stack[:-1]  # Remove the last element
        context_stack.set(new_stack)
        
        # Remove all sinks created by this context
        for sink_id in self.sink_ids:
            try:
                logger.remove(sink_id)
            except Exception:
                pass  # Sink may already be removed
                
    def _create_viz_sink(self) -> None:
        """Create a plain text sink for visualization output.
        
        Creates a text file with a custom format that excludes log formatting.
        """
        try:
            # Get configuration for this context
            log_level = self.config.get_context_value(self.context_name, "level", DEFAULT_LOG_LEVEL)
            
            # Process viz filename template
            filename_template = self.config.get_value("general_defaults.viz_context_default_filename",
                                                     "%[viz_name].txt")
            filename = filename_template.replace("%[viz_name]", self.viz_name.replace(':', '_').replace(' ', '_'))
            
            # Ensure .txt extension
            if not filename.endswith(".txt"):
                filename += ".txt"
                
            # Get viz directory from global config (which has the timestamped path)
            viz_dir = global_config.get_value("general_defaults.log_dir_viz")
            # Normalize path separators for the OS
            viz_dir = os.path.normpath(viz_dir)
            filepath = os.path.join(viz_dir, filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Define a filter function that only allows messages when this context is active
            context_name = self.context_name  # Capture for closure
            def context_filter(record):
                current_context = get_current_context()
                return current_context == context_name
            
            # Create sink with minimal formatting - just the message
            sink_id = logger.add(
                sink=filepath,
                level=log_level,
                format="{message}",  # Plain text, no timestamp or level
                filter=context_filter,  # Only log when this context is active
                enqueue=True,  # Use queue for thread safety
            )
            
            # Track the sink
            self.sink_ids.append(sink_id)
            
        except Exception as e:
            logger.error(f"Failed to create viz sink for {self.viz_name}: {e}")
            self.sink_creation_failed = True

### 4. Logging System Initialization

# Filter function to exclude messages from visualization contexts
def non_viz_context_filter(record):
    """Filter that excludes messages from visualization contexts.
    
    Args:
        record: Log record
        
    Returns:
        True if the message should be logged, False otherwise
    """
    current_context = get_current_context()
    # Only allow messages that are NOT from visualization contexts
    return current_context is None or not current_context.startswith("viz_")

def initialize_logging(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Initialize the logging system with the specified configuration.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    global global_config
    
    # Load configuration
    global_config = LoggingConfig(config_path)
    
    # Configure Loguru to add context to all records
    logger.configure(patcher=add_context_to_record)
    
    # Get configuration values
    console_enabled = global_config.get_value("general_defaults.console_enabled", True)
    console_level = global_config.get_stream_value("console", "level", 
                                                 global_config.get_value("general_defaults.level", DEFAULT_LOG_LEVEL))

    console_format = global_config.get_stream_value("console", "log_msg_format", 
                                                  global_config.get_value("general_defaults.log_msg_format", DEFAULT_FORMAT))
    
    file_enabled = global_config.get_value("general_defaults.file_enabled", True)
    error_file_enabled = global_config.get_value("general_defaults.error_file_enabled", True)
    run_file_enabled = global_config.get_value("general_defaults.run_file_enabled", True)
    
    # Get log directories and resolve them relative to config directory
    log_dir_run = global_config._resolve_path(global_config.get_value("general_defaults.log_dir_run", "logs/run"))
    log_dir_viz = global_config.get_value("general_defaults.log_dir_viz", "logs/viz")
    
    # Add datetime stamp to run directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_dir_run_with_timestamp = f"{log_dir_run}_{timestamp}"
    
    # Set viz directory as subdirectory of run directory with timestamp
    log_dir_viz_with_timestamp = os.path.join(log_dir_run_with_timestamp, log_dir_viz)
    
    # Update global config with timestamped paths
    global_config.config["general_defaults"]["log_dir_run"] = log_dir_run_with_timestamp
    global_config.config["general_defaults"]["log_dir_viz"] = log_dir_viz_with_timestamp
    
    # Normalize paths
    log_dir_run_with_timestamp = os.path.normpath(log_dir_run_with_timestamp)
    log_dir_viz_with_timestamp = os.path.normpath(log_dir_viz_with_timestamp)
    
    # Create log directories if they don't exist
    os.makedirs(log_dir_run_with_timestamp, exist_ok=True)
    os.makedirs(log_dir_viz_with_timestamp, exist_ok=True)
    
    # Configure console logger
    if console_enabled:
        logger.remove(0)  # Remove default handler
        logger.add(
            sink=sys.stderr,
            level=console_level,
            format=console_format,
            colorize=True,
            backtrace=True,
            diagnose=True,
            filter=non_viz_context_filter,  # Exclude viz contexts
        )
    
    # Configure file loggers
    if file_enabled:
        # Run log - records all messages at specified level
        if run_file_enabled:
            run_log_path = os.path.join(log_dir_run_with_timestamp, "run.log")
            run_file_level = global_config.get_stream_value("run_file", "level", 
                                                           global_config.get_value("general_defaults.level", DEFAULT_LOG_LEVEL))
            run_file_format = global_config.get_stream_value("run_file", "log_msg_format", 
                                                           global_config.get_value("general_defaults.log_msg_format", DEFAULT_FORMAT))
            
            logger.add(
                sink=run_log_path,
                level=run_file_level,
                format=strip_ansi_escape_sequences(run_file_format),
                filter=non_viz_context_filter,  # Exclude viz contexts
                enqueue=True,  # Use queue for thread safety
                backtrace=True,  # Include traceback in errors
                diagnose=True,  # Include variables in traceback
            )
        
        # Error log - records only warnings and errors
        if error_file_enabled:
            error_log_path = os.path.join(log_dir_run_with_timestamp, "errors.log")
            error_file_level = global_config.get_stream_value("error_file", "level", 
                                                            global_config.get_value("general_defaults.level", "WARNING"))
            error_file_format = global_config.get_stream_value("error_file", "log_msg_format", 
                                                             global_config.get_value("general_defaults.log_msg_format", DEFAULT_FORMAT))
            
            logger.add(
                sink=error_log_path,
                level=error_file_level,
                format=strip_ansi_escape_sequences(error_file_format),
                filter=lambda record: non_viz_context_filter(record) and record["level"].name in ["WARNING", "ERROR", "CRITICAL"],  # Combine filters
                enqueue=True,  # Use queue for thread safety
                backtrace=True,  # Include traceback in errors
                diagnose=True,  # Include variables in traceback
            )

# Create context-aware logger
def context_aware_logger(original_logger):
    """Create a context-aware logger that adds context prefixes to messages.
    
    Args:
        original_logger: Original logger instance
        
    Returns:
        A wrapper function that adds context information to messages
    """
    
    class ContextAwareLogger:
        def __getattr__(self, name):
            original_method = getattr(original_logger, name)
            
            if callable(original_method):
                def wrapped_method(*args, **kwargs):
                    if args and isinstance(args[0], str):
                        # First argument is the message, add context prefix
                        args = (with_context_prefix(args[0]),) + args[1:]
                    return original_method(*args, **kwargs)
                
                return wrapped_method
            else:
                return original_method
    
    return ContextAwareLogger()

# Create context-aware logger
context_logger = context_aware_logger(logger)

# Expose public interface
__all__ = [
    'logger',
    'context_logger',
    'LoggingContext',
    'VizContext',
    'initialize_logging',
    'get_current_context',
]
