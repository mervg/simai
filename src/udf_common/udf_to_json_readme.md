# UDF to JSON Export Module

## Overview

The UDF to JSON Export module (`udf_to_json.py`) is a flexible utility for exporting User-Defined Format (UDF) records from an SQLite database to structured JSON files. It's designed with a dual-use architecture, functioning both as a standalone command-line tool and as an integrated library component within the DM_tasks processing pipeline.

This module dynamically analyzes UDF records to determine relationships between different record types (e.g., tasks and task instances) and creates an appropriate hierarchical or flat JSON structure based on the detected relationships.

## Features

- **Flexible Record Selection**: Filter UDF records using standard SQLite WHERE clauses
- **Dynamic Structure Analysis**: Automatically detect relationships between record types
- **Hierarchical JSON Output**: Create nested structures for related records
- **Comprehensive Logging**: Detailed logging with proper context for each processing step
- **Robust Error Handling**: Graceful handling of database, file I/O, and validation errors
- **Dual-Mode Operation**: Functions both as a CLI tool and as a library component

## Installation

The module is part of the AI Task Analyzer project and requires no additional installation beyond the project dependencies:

- Python 3.6+
- sqlite3
- json
- os
- argparse
- re
- sys

## Configuration

The module uses several default configuration values that can be overridden:

```python
# Default SQLite DB path
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sqlite_db')
SQLITE_DB_FILENAME = os.path.join(SQLITE_DB_PATH, 'UDF.db')

# Default JSON output directory
JSON_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'UDF_json')

# Logging configuration
LOGGING_CONFIG_FILENAME = 'logger/udf2json_log_config.yaml'
```

## Usage

### As a Command-Line Tool

The module can be run directly from the command line with various options:

```bash
# Basic usage with WHERE clause
python udf_to_json.py --where "udf_entity_id LIKE '%2-637%'"

# Specify custom database file
python udf_to_json.py --where "udf_entity_id = '2-637'" --db "../custom/path/database.db"

# Specify custom output directory
python udf_to_json.py --where "udf_entity_id LIKE '%2-637%'" --output "../custom/output"

# Specify custom logging configuration
python udf_to_json.py --where "udf_entity_id LIKE '%2-637%'" --log-config "../custom/log_config.yaml"

# Get help
python udf_to_json.py --help
```

Command-line arguments:

- `--where`: SQLite WHERE clause for filtering UDF records (required)
- `--db`: Path to SQLite database file (optional)
- `--output`: Directory for JSON output files (optional)
- `--log-config`: Path to logging configuration file (optional)

### As a Library Component

The module is designed to be used directly within the DM_tasks.py process flow as step S1500. Here's how it's integrated:

```python
# S1500 - UDF to JSON Export
logger.key("S1500 - UDF to JSON Export Starting...")

where_clause = "udf_entity_id LIKE '%2-637%'"

try:
    # Export task with ID 637 and all related instances
    with LoggingContext("S1500 - UDF to JSON Export"):
        
        logger.info("S1510: Using Existing UDF connection")
        
        # Validate WHERE clause
        with LoggingContext("S1520 - UDF WHERE clause Validation"):
            logger.debug(f"WHERE clause: {where_clause}")
            if not where_clause or not where_clause.strip():
                logger.error("WHERE clause is empty or invalid")
                raise ValueError("WHERE clause is empty or invalid")
            logger.info(f"WHERE clause validated: {where_clause}")
        logger.info(f"S1520: OK: WHERE clause validated: {where_clause}")
        
        try:
            with LoggingContext("S1530 - Read UDF as JSON"):
                udf_json_data = read_udf_as_json(cursor, where_clause)
                logger.key("S1530: OK: UDF to JSON Read completed successfully exported")
        except Exception as e:
            logger.error(f"S1530 - ERROR: Failed to read UDF data: {str(e)}")
            raise
        
        # Write JSON to file
        write_json_dir = WRITE_JSON_DIR
        try:
            with LoggingContext("S1540 - Write UDF to JSON"):
                output_json_file = write_udf_json_to_file(udf_json_data, where_clause, write_json_dir)
                logger.key(f"S1540: OK: UDF to JSON Write completed successfully - {output_json_file}")
        except Exception as e:
            logger.error(f"S1540 - ERROR: Failed to write JSON file: {str(e)}")
        
    logger.key("S1500 - UDF to JSON Export completed successfully")
except Exception as e:
    logger.error(f"S1500 - ERROR: UDF to JSON Export failed: {str(e)}")
```

## Technical Implementation

The module consists of several key components that work together to export UDF records to JSON:

### Core Functions

#### 1. `read_udf_as_json(cursor, where_clause)`

This is the main function for retrieving and structuring UDF records:

```python
def read_udf_as_json(cursor: sqlite3.Cursor, where_clause: str) -> Dict[str, Any]:
    """
    Retrieves UDF records based on WHERE clause and transforms them into structured JSON object.
    """
    # Implementation details...
```

- Validates the WHERE clause
- Executes the SQL query
- Fetches records and column information
- Processes records into structured JSON
- Discovers relationships between record types
- Creates appropriate JSON structure (hierarchical or flat)

#### 2. `process_json_structure(records, columns)`

Processes raw database records into structured dictionaries:

```python
def process_json_structure(records: List[Tuple], columns: List[str]) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Processes raw database records into structured dictionaries.
    """
    # Implementation details...
```

- Converts raw records to dictionaries
- Parses JSON string fields (meta_dates, meta_product, etc.)
- Groups records by udf_type
- Returns structured data and type groups

#### 3. `discover_relationships(type_groups)`

Analyzes entity ID patterns to discover relationships:

```python
def discover_relationships(type_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    Discover relationships between different udf_types based on entity_id patterns.
    """
    # Implementation details...
```

- Analyzes entity_id patterns across different UDF types
- Identifies parent-child relationships
- Returns mapping of child types to parent types

#### 4. `build_task_hierarchy(type_groups)`

Builds hierarchical structure for tasks and instances:

```python
def build_task_hierarchy(type_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Build a hierarchical structure for tasks and task instances.
    """
    # Implementation details...
```

- Organizes tasks and their instances into a nested structure
- Creates a hierarchical JSON object
- Handles cases where tasks have multiple instances

#### 5. `write_udf_json_to_file(json_data, where_clause, output_dir)`

Writes JSON data to a file:

```python
def write_udf_json_to_file(json_data: Dict[str, Any], where_clause: str, output_dir: Optional[str] = None) -> str:
    """
    Writes JSON data to a file with name based on where_clause.
    """
    # Implementation details...
```

- Creates output directory if it doesn't exist
- Sanitizes WHERE clause for use in filename
- Writes JSON with proper indentation
- Returns path to the created file

### Process Flow

The module follows a structured process flow:

1. **S1510 - Database Connection**: Establish or reuse SQLite connection
2. **S1520 - WHERE Clause Validation**: Validate the provided WHERE clause
3. **S1530 - Read UDF as JSON**: Retrieve and structure UDF records
   - **S1531 - Query Execution**: Execute SQL query with WHERE clause
   - **S1532 - Record Fetching**: Fetch records and column information
   - **S1533 - Process JSON Structure**: Convert records to structured dictionaries
   - **S1534 - Discover Relationships**: Analyze entity ID patterns
   - **S1535 - Build Hierarchy**: Create hierarchical structure if appropriate
4. **S1540 - Write to File**: Write JSON data to file
5. **S1550 - Close Connection**: Close SQLite connection (standalone mode only)

## Output JSON Structure

The module produces two types of JSON structures:

### Flat Structure (Default)

```json
{
  "records": [
    {
      "udf_id": 1,
      "udf_type": "tasksdm_task",
      "udf_entity_id": "2-637",
      "meta_dates": { "created": "2023-01-01", "modified": "2023-01-02" },
      "meta_product": { "name": "Product A", "version": "1.0" },
      ...
    },
    {
      "udf_id": 2,
      "udf_type": "tasksdm_task_instance",
      "udf_entity_id": "2-637-145",
      "meta_dates": { "created": "2023-01-03", "modified": "2023-01-04" },
      ...
    }
  ]
}
```

### Hierarchical Structure (For Tasks and Instances)

```json
{
  "tasks": [
    {
      "task": {
        "udf_id": 1,
        "udf_type": "tasksdm_task",
        "udf_entity_id": "2-637",
        "meta_dates": { "created": "2023-01-01", "modified": "2023-01-02" },
        ...
      },
      "instances": [
        {
          "udf_id": 2,
          "udf_type": "tasksdm_task_instance",
          "udf_entity_id": "2-637-145",
          "meta_dates": { "created": "2023-01-03", "modified": "2023-01-04" },
          ...
        }
      ]
    }
  ]
}
```

## Error Handling

The module implements comprehensive error handling:

- **Database Errors**: Handles SQLite connection and query errors
- **Validation Errors**: Validates WHERE clause and other inputs
- **JSON Parsing Errors**: Handles errors when parsing JSON fields
- **File I/O Errors**: Handles errors when creating directories or writing files

All errors are properly logged with context information using the LoggingContext system.

## Logging

The module uses a structured logging system with LoggingContext:

```python
with LoggingContext("S1530 - Read UDF as JSON"):
    udf_json_data = read_udf_as_json(cursor, where_clause)
```

Logging configuration can be customized via the `udf2json_log_config.yaml` file.

## Examples

### Example 1: Export a specific task and all its instances

```bash
python udf_to_json.py --where "udf_entity_id LIKE '%2-637%'"
```

This will export the task with ID 637 and all its instances to a JSON file.

### Example 2: Export a single task record

```bash
python udf_to_json.py --where "udf_entity_id = '2-637'"
```

This will export only the task with ID 637 to a JSON file.

### Example 3: Export a single task instance record

```bash
python udf_to_json.py --where "udf_entity_id = '2-637-145'"
```

This will export only the task instance with ID 145 for task 637 to a JSON file.

### Example 4: Export all records

```bash
python udf_to_json.py --where "1=1"
```

This will export all UDF records to a JSON file.

## Best Practices

1. **Use Specific WHERE Clauses**: Be as specific as possible with WHERE clauses to limit the number of records processed
2. **Handle Large Results**: For large result sets, consider using more specific WHERE clauses
3. **Check Output Files**: Always verify the output JSON files for correctness
4. **Use Proper Logging**: Configure logging appropriately for your environment
5. **Error Handling**: Always handle errors properly when using the module as a library component

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Ensure the database file exists and is accessible
   - Check file permissions

2. **WHERE Clause Errors**:
   - Ensure the WHERE clause is valid SQLite syntax
   - Avoid SQL injection by validating user input

3. **Output Directory Issues**:
   - Ensure the output directory is writable
   - Check file permissions

4. **JSON Parsing Errors**:
   - Check that JSON fields in the database are valid JSON
   - Handle NULL values appropriately

## Contributing

When contributing to this module, please follow these guidelines:

1. Maintain the dual-use architecture (CLI and library component)
2. Follow the established logging and error handling patterns
3. Update documentation when adding new features
4. Add appropriate tests for new functionality

## License

This module is part of the AI Task Analyzer project and is subject to the same license terms.

## Related Documentation

- [Revised Process Flow](Revised_Process_Flow.md): Detailed documentation of the process flow
- [Project Brief](project_docs/PROJECT%20BRIEF.md): Overview of the project
- [Implementation Guidelines](Implementation/README.md): Implementation guidelines for the project
