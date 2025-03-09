# Advanced Logging System Documentation

## Overview

The AI Task Analyzer logging system is a sophisticated, context-aware logging solution built on top of [Loguru](https://github.com/Delgan/loguru). It provides enhanced logging capabilities with minimal code overhead and maximum flexibility.

Key features:
- Multiple output streams (console, files, visualization)
- Context-aware logging with automatic file routing
- Hierarchical configuration via YAML
- Custom log levels
- Plain text visualization output
- Thread-safe logging

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Logging Contexts](#logging-contexts)
4. [Configuration](#configuration)
5. [Custom Log Levels](#custom-log-levels)
6. [Usage Examples](#usage-examples)
7. [Advanced Features](#advanced-features)

## Quick Start

### Basic Setup

```python
from src.logger import logger, initialize_logging

# Initialize with default configuration
initialize_logging()

# Basic logging at different levels
logger.debug("Debug message")
logger.info("Information message")
logger.key("Important information")  # Custom level between INFO and WARNING
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")
```

### Context-Specific Logging

```python
from src.logger import logger, LoggingContext

# Log to both console and a process-specific file
with LoggingContext("data_processing"):
    logger.info("Processing data...")
    logger.key("Found 10 records")
    
    # Nested context
    with LoggingContext("validation"):
        logger.info("Validating records...")
```

### Visualization Output

```python
from src.logger import logger, VizContext

# Create plain text output for visualization
with VizContext("histogram"):
    logger.info("# Frequency Distribution")
    logger.info("Value | Count")
    logger.info("------|------")
    logger.info("A     | 10")
    logger.info("B     | 25")
    logger.info("C     | 15")
```

## Core Components

### Logger

The `logger` object is the main entry point for logging. It's an enhanced version of Loguru's logger with additional features:

- Context awareness
- Custom log levels
- Multiple output sinks with filtering

### Context Managers

The system provides two primary context managers:

- **LoggingContext**: For process-specific logging that routes messages to dedicated log files
- **VizContext**: For visualization output that writes plain text to .txt files

### Configuration Manager

The `LoggingConfig` class handles loading and accessing configuration from YAML files with hierarchical overrides.

## Logging Contexts

### Process Context

Process contexts create dedicated log files for specific processes or tasks:

```python
with LoggingContext("data_import"):
    logger.info("Importing data...")
    # Logs to both console and data_import.log
```

Process contexts:
- Persist their sinks for reuse
- Filter messages based on the active context
- Support hierarchical configuration
- Support nested contexts

### Visualization Context

Visualization contexts create plain text output files without log formatting:

```python
with VizContext("chart_data"):
    logger.info("X,Y")
    logger.info("1,10")
    logger.info("2,20")
    # Creates chart_data.txt with just the message content
```

Visualization contexts:
- Create plain text output (no timestamps, levels, etc.)
- Clean up sinks when exiting
- Filter messages to only appear in their specific output
- Are automatically excluded from main console and file logs

## Configuration

The system uses a hierarchical YAML configuration:

1. Context-specific settings (highest priority)
2. Stream-specific settings
3. General defaults (lowest priority)

### Default Configuration File

```yaml
# General defaults
general_defaults:
  level: "INFO"
  log_msg_format: "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
  log_dir_run: "../logs/run"
  log_dir_viz: "./viz"
  console_enabled: true
  file_enabled: true
  error_file_enabled: true
  run_file_enabled: true
  process_context_default_filename: "%[context_name]"
  viz_context_default_filename: "%[viz_name].txt"

# Stream-specific settings
streams:
  console:
    level: "INFO"
    log_msg_format: "| <level>{level: <8}</level> | <green>{extra[context]: <15}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
  
  run_file:
    level: "DEBUG"
  
  context_file:
    level: "DEBUG"
  
  error_file:
    level: "WARNING"

# Context-specific settings
logging_contexts:
  validation:
    level: "KEY"
    log_msg_format: "<custom format>"
```

### Configuration Path

You can specify a custom configuration path when initializing the logging system:

```python
initialize_logging("/path/to/custom/logging_config.yaml")
```

### Format Strings

The system supports Loguru's format strings with added context awareness:

- `{extra[context]}`: Name of the current context
- All standard Loguru format fields

## Custom Log Levels

The system adds a `KEY` log level between `INFO` and `WARNING`:

```python
logger.key("Important message")  # Level between INFO and WARNING
```

This is useful for information that's more important than regular INFO but not a WARNING.

## Usage Examples

### Basic Logging

```python
# Import the logger
from src.logger import logger, initialize_logging

# Initialize logging system
initialize_logging()

# Log messages at different levels
logger.debug("Debug message")
logger.info("Information message")
logger.key("Important information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical failure")
```

### Process-Specific Logging

```python
from src.logger import logger, LoggingContext

# Analyze data with dedicated log file
with LoggingContext("data_analysis"):
    logger.info("Starting analysis...")
    
    # Process data
    data = [1, 2, 3, 4, 5]
    logger.info(f"Processing {len(data)} records")
    
    # Log key findings
    mean = sum(data) / len(data)
    logger.key(f"Mean value: {mean}")
    
    # Log warnings
    if len(data) < 10:
        logger.warning("Small sample size, results may be unreliable")
```

### Nested Contexts

```python
from src.logger import logger, LoggingContext

with LoggingContext("main_process"):
    logger.info("Starting main process")
    
    # First sub-process
    with LoggingContext("sub_process_1"):
        logger.info("Running sub-process 1")
        
    # Second sub-process
    with LoggingContext("sub_process_2"):
        logger.info("Running sub-process 2")
        
    logger.info("Main process completed")
```

### Visualization Output

```python
from src.logger import logger, VizContext

# Create a table visualization
with VizContext("data_table"):
    logger.info("| ID | Name  | Value |")
    logger.info("|----+-------+-------|")
    logger.info("| 1  | Alpha | 10.5  |")
    logger.info("| 2  | Beta  | 20.3  |")
    logger.info("| 3  | Gamma | 15.7  |")

# Create a simple chart
with VizContext("line_chart"):
    logger.info("X,Y")
    for x in range(10):
        y = x * x
        logger.info(f"{x},{y}")
```

### Error Handling with Context

```python
from src.logger import logger, LoggingContext, VizContext

def process_data():
    with LoggingContext("data_processor"):
        try:
            logger.info("Processing data...")
            # Simulate error
            result = 10 / 0
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.exception("Full traceback:")
            return False
    return True

# Visualize error statistics
with VizContext("error_stats"):
    logger.info("Error Type | Count")
    logger.info("-----------|------")
    logger.info("ValueError | 5")
    logger.info("TypeError  | 2")
    logger.info("ZeroDivisionError | 1")
```

## Advanced Features

### Context-Aware Logger

The system provides a `context_logger` that automatically adds context prefixes to messages:

```python
from src.logger import context_logger

# Regular logging
context_logger.info("This message gets a context prefix")
```

### Context Name in Format Strings

You can include the context name in your log formats:

```yaml
log_msg_format: "{time} | {extra[context]} | {level} | {message}"
```

### Log File Organization

The system automatically organizes log files:
- Creates timestamped run directories
- Places visualization files in a subdirectory
- Handles file naming with templates

### Custom Filters

The system uses filters to route messages to the appropriate sinks:
- Context-specific filtering
- Visualization context exclusion from main logs
- Level-based filtering

### Thread Safety

All logging operations are thread-safe with Loguru's built-in `enqueue` feature.

### ANSI Color Stripping

The system automatically strips ANSI color codes from file output while preserving colored console output.

## Conclusion

This advanced logging system provides a flexible and powerful solution for logging in the AI Task Analyzer project. By leveraging contexts, hierarchical configuration, and specialized output formats, it simplifies logging while enhancing its capabilities.
