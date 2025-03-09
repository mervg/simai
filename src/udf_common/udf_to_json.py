import sqlite3
import json
import os
import argparse
import re
import sys
from typing import Dict, List, Tuple, Any, Optional

# Handle imports based on how the module is being used
if __name__ == "__main__":
    # When run directly as a script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from logger.logger import LoggingContext, logger, initialize_logging
    from db_sqlite_ops import sqlite_db_open, sqlite_db_close
    LOGGING_CONFIG_FILENAME = '../logger/udf2json_log_config.yaml'
else:
    # When imported from another module
    try:
        # Try relative imports first (when imported from a package)
        from ..logger.logger import LoggingContext, logger, initialize_logging
        from ..db_sqlite_ops import sqlite_db_open, sqlite_db_close
        LOGGING_CONFIG_FILENAME = '../../logger/udf2json_log_config.yaml'
    except ImportError:
        # Fall back to absolute imports (when imported from a script)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from logger.logger import LoggingContext, logger, initialize_logging
        from db_sqlite_ops import sqlite_db_open, sqlite_db_close
        LOGGING_CONFIG_FILENAME = '../logger/udf2json_log_config.yaml'

# Default SQLite DB path
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sqlite_db')
SQLITE_DB_FILENAME = os.path.join(SQLITE_DB_PATH, 'UDF.db')

# Default JSON output directory
JSON_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'UDF_json')


def read_udf_as_json(cursor: sqlite3.Cursor, where_clause: str) -> Dict[str, Any]:
    """
    Retrieves UDF records based on WHERE clause and transforms them into structured JSON object.
    
    This function queries the database using the provided WHERE clause, processes the 
    retrieved records into a standardized JSON structure, and identifies relationships 
    between different UDF record types to create either a hierarchical or flat structure.
    
    Args:
        cursor (sqlite3.Cursor): SQLite database cursor for executing queries
        where_clause (str): SQL WHERE clause for filtering UDF records (without the "WHERE" keyword)
        
    Returns:
        Dict[str, Any]: Structured JSON data with appropriate relationships between records
        
    Raises:
        ValueError: If where_clause is empty or invalid
        sqlite3.Error: If there's an error executing the SQL query or processing results
    """
    if not where_clause or not isinstance(where_clause, str):
        raise ValueError("WHERE clause must be a non-empty string")
    
    try:
        # Construct and execute the query
        with LoggingContext("S1531 - UDF Query Execution"):
            try:
                logger.debug(f"Executing query: {where_clause}")
                query = f"SELECT * FROM UDF WHERE {where_clause}"
                cursor.execute(query)
                logger.debug("OK: Query executed")
            except sqlite3.Error as e:
                logger.error(f"SQLite error during UDF query: {str(e)}")
                raise
        logger.info("S1531 OK: UDF Queried")

        # Get column names and fetch all records
        with LoggingContext("S1532 - UDF Record Fetch and Processing"):
            columns = [description[0] for description in cursor.description]
            records = cursor.fetchall()
                
            record_count = len(records)
            logger.info(f"Retrieved {record_count} UDF records")
                
            if record_count == 0:
                logger.warning(f"No UDF records found matching: {where_clause}")
                return {"records": []}
        logger.info("S1532 OK: UDF Records fetched")
            
        # Process records into structured format
        with LoggingContext("S1533 - UDF Process JSON structure"):
            structured_data, type_groups = process_json_structure(records, columns)
        logger.info("S1533 OK: UDF Processed JSON structure")

        # Discover relationships between types
        with LoggingContext("S1534 - UDF Discover Relationships in JSON"):
            relationships = discover_relationships(type_groups)
        logger.info("S1534 OK: UDF Discovered Relationships in JSON")
        
        # If we have task and task_instance types with relationships, create a nested structure
        if 'tasksdm_task' in type_groups and 'tasksdm_task_instance' in type_groups and relationships:
            with LoggingContext("S1535 - UDF Tasks Specific hierarchy processing"):
                structured_data = build_task_hierarchy(type_groups)
            logger.info("S1535 OK: UDF Tasks Specific hierarchy processed")
        else:
            logger.info("S1535 OK: No specific hierarchy processing.")
        
        logger.info("S1530 OK: Created structured JSON")
        return structured_data
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error during UDF data retrieval: {str(e)}")
        raise

def process_json_structure(records: List[Tuple], columns: List[str]) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Analyzes UDF records and determines appropriate JSON structure.
    
    Processes raw database records into structured dictionaries by:
    1. Converting raw records into dictionaries with proper JSON parsing
    2. Grouping records by udf_type for relationship analysis
    3. Preparing data for hierarchical or flat structure creation
    
    Args:
        records (List[Tuple]): Raw records from SQLite query
        columns (List[str]): Column names for the records
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]: 
            - Structured JSON data dictionary with 'records' key
            - Records grouped by udf_type for relationship analysis
    """
    # Convert records to dictionaries
    logger.debug("Converting records to dictionaries...")
    record_dicts = []
    for record in records:
        record_dict = {}
        # Log the entity ID for debugging
        entity_id = record[columns.index('udf_entity_id')] if 'udf_entity_id' in columns else 'unknown'
        logger.debug(f"Processing record with entity_id: {entity_id}")
        for i, col in enumerate(columns):
            # Skip NULL values
            if record[i] is None:
                logger.debug(f"Skipping NULL value for column: {col}")
                continue
                
            # Parse JSON fields
            if col in ["meta_dates", "meta_product", "meta_store", "meta_user", 
                      "meta_tags", "meta_context", "meta_result"]:
                #logger.debug(f"Parsing JSON field: {col}")
                try:
                    record_dict[col] = json.loads(record[i])
                    logger.debug(f"OK: meta JSON parsed for column: {col}")
                except json.JSONDecodeError:
                    record_dict[col] = record[i]
                    logger.debug(f"OK: non-meta field added for column: {col}")
            else:
                record_dict[col] = record[i]
                logger.debug(f"OK: non-meta field added for column: {col}")
        record_dicts.append(record_dict)
    logger.debug("OK: Records converted to dictionaries")
    
    # Group by udf_type
    logger.debug("Grouping records by udf_type...")
    type_groups = {}
    for record in record_dicts:
        udf_type = record.get('udf_type')
        if not udf_type:
            logger.debug("Skipping record without udf_type")
            continue
            
        if udf_type not in type_groups:
            logger.debug(f"Adding new udf_type: {udf_type}")
            type_groups[udf_type] = []
        
        logger.debug(f"Adding record to {udf_type} group")
        type_groups[udf_type].append(record)
    
    logger.debug("OK: Records grouped by udf_type")
    
    return {"records": record_dicts}, type_groups

def discover_relationships(type_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    Discover relationships between different udf_types based on entity_id patterns.
    
    Analyzes entity_id patterns across different UDF record types to determine
    hierarchical relationships. For example, if one record has entity_id "A-123" 
    and another has "A-123-456", the second is likely a child of the first.
    
    Args:
        type_groups (Dict[str, List[Dict[str, Any]]]): Records grouped by udf_type
        
    Returns:
        Dict[str, str]: Mapping of child type to parent type where keys are child
                        UDF types and values are their corresponding parent UDF types
    """
    relationships = {}
    
    # If we have less than 2 types, no relationships to discover
    if len(type_groups) < 2:
        logger.debug("Not enough types to discover relationships")
        return relationships
    
    # Check each pair of types for potential relationships
    logger.debug("Discovering relationships between types...")
    for child_type, child_records in type_groups.items():
        for parent_type, parent_records in type_groups.items():
            if child_type == parent_type:
                logger.debug(f"Skipping self relationship: {child_type}")
                continue
                
            # Check if child entity_ids contain parent entity_ids
            logger.debug(f"Checking relationships between {child_type} and {parent_type}")
            for child_record in child_records:
                child_entity_id = child_record.get('udf_entity_id', '')
                
                for parent_record in parent_records:
                    parent_entity_id = parent_record.get('udf_entity_id', '')
                    
                    # If child entity_id starts with parent entity_id and has additional parts
                    logger.debug(f"Checking relationship between {child_type} and {parent_type}")
                    if (child_entity_id.startswith(parent_entity_id) and 
                        child_entity_id != parent_entity_id and
                        len(child_entity_id.split('-')) > len(parent_entity_id.split('-'))):
                        relationships[child_type] = parent_type
                        logger.debug(f"OK: Discovered relationship: {child_type} -> {parent_type}")
                        break
                    logger.debug(f"OK: No relationship found between {child_type} and {parent_type}")
                
                if child_type in relationships:
                    logger.debug(f"OK: Discovered relationship: {child_type} -> {parent_type}")
                    break
    
    logger.debug(f"Discovered relationships: {relationships}")
    return relationships

def build_task_hierarchy(type_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Build a hierarchical structure for tasks and task instances.
    
    This function organizes UDF records into a nested structure where tasks contain
    their associated instances. It identifies task-instance relationships based on
    entity ID patterns, where an instance's entity ID starts with its parent task's
    entity ID followed by an additional identifier.
    
    Args:
        type_groups (Dict[str, List[Dict[str, Any]]]): Records grouped by udf_type
        
    Returns:
        Dict[str, Any]: Hierarchical JSON structure with tasks and their instances
    """
    # Create a structure with tasks and their instances
    tasks = type_groups.get('tasksdm_task', [])
    instances = type_groups.get('tasksdm_task_instance', [])
    
    logger.debug("Building task hierarchy...")
    result = {
        "tasks": []
    }
    
    # Process each task
    logger.debug("Processing tasks...")
    for task in tasks:
        task_entity_id = task.get('udf_entity_id', '')
        task_with_instances = {
            "task": task,
            "instances": []
        }
        
        # Find instances for this task
        logger.debug("Finding instances for task...")
        for instance in instances:
            instance_entity_id = instance.get('udf_entity_id', '')
            # Check if instance belongs to this task
            if instance_entity_id.startswith(task_entity_id + '-'):
                logger.debug(f"OK: Instance {instance_entity_id} belongs to task {task_entity_id}")
                task_with_instances["instances"].append(instance)
            else:
                logger.debug(f"OK: Instance {instance_entity_id} does not belong to task {task_entity_id}")
        result["tasks"].append(task_with_instances)
    
    logger.debug("OK: Task hierarchy built successfully")
    return result

def write_udf_json_to_file(json_data: Dict[str, Any], where_clause: str, output_dir: Optional[str] = None) -> str:
    """
    Writes JSON data to a file with name based on where_clause.
    
    Args:
        json_data (Dict[str, Any]): Structured JSON data to write
        where_clause (str): WHERE clause used to query the data
        output_dir (Optional[str]): Directory for output file, defaults to '../output'
        
    Returns:
        str: Path to the written file
        
    Raises:
        IOError: If there's an error creating the directory or writing the file
    """
    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        raise IOError(f"Failed to create output directory: {str(e)}")
        
    # Sanitize where_clause for filename
    safe_where = sanitize_where_for_filename(where_clause)
    filename = f"udf_{safe_where}.json"
    file_path = os.path.join(output_dir, filename)
        
    # Write JSON to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON data written to: {file_path}")
        return file_path
    except IOError as e:
        logger.error(f"Failed to write JSON file: {str(e)}")
        raise IOError(f"Failed to write JSON file: {str(e)}")

def sanitize_where_for_filename(where_clause: str) -> str:
    """
    Sanitizes WHERE clause for use in filename.
    
    Args:
        where_clause (str): WHERE clause to sanitize
        
    Returns:
        str: Sanitized string safe for use in filename
    """
    # Replace unsafe characters
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', where_clause)
    # Replace spaces and quotes
    safe_name = re.sub(r'[\s\'"]', '_', safe_name)
    # Replace multiple underscores with single
    safe_name = re.sub(r'_+', '_', safe_name)
    # Limit length
    if len(safe_name) > 100:
        safe_name = safe_name[:97] + '...'
    return safe_name

def main():
    """
    Command-line interface for UDF to JSON export.
    
    Parses command line arguments, initializes logging, establishes database 
    connection, validates WHERE clause, reads UDF data as JSON, writes JSON to 
    file, and handles errors appropriately.
    
    Arguments are processed for:
    - Database file path
    - WHERE clause for filtering records
    - Output directory for JSON files
    - Logging configuration file
    
    Returns:
        int: 0 for success, 1 for error (as exit code)
    
    Raises:
        Various exceptions that are caught, logged, and then re-raised
    """
    parser = argparse.ArgumentParser(description='Export UDF records to JSON based on WHERE clause')
    parser.add_argument('--db', '-d', default=SQLITE_DB_FILENAME, help=f'Optional SQLite DB filename, default: {SQLITE_DB_FILENAME}')
    parser.add_argument('--where', '-w', required=True, help='WHERE clause for SQLite query, WHERE True for all records, WHERE udf_entity_id = "2-637" for task, WHERE udf_entity_id = "2-637-145" for task instance, WHERE udf_entity_id like "%%2-637%%" for task and all instances')
    parser.add_argument('--output-dir', '-o', default=JSON_OUTPUT_DIR, help=f'Optional Directory for JSON output, default: {JSON_OUTPUT_DIR}')
    parser.add_argument('--logconfig', '-lc', default=LOGGING_CONFIG_FILENAME, help=f'Optional Logging configuration file, default: {LOGGING_CONFIG_FILENAME}')
    
    args = parser.parse_args()
    db_path = args.db
    write_json_dir = args.output_dir
    where_clause = args.where
    logging_config = args.logconfig

    # Initialize logging - setting the path to log files, e.g. log_dir_run: "../../output/logs/udf2json" 
    initialize_logging(logging_config)

    logger.key(f"UDF to JSON Export, WHERE clause: {where_clause}, Output directory: {write_json_dir}, using DB {db_path}, logging config: {logging_config}")
    logger.key(f"For detailed process flow logging, see logs directory specified in {logging_config}")

    # Track if we need to close the connection
    close_connection = False

    try:
        with LoggingContext("S1500 - UDF to JSON Export"):

            # Set up database connection if not provided
            logger.key("S1510 - UDF Database Connection Opening...")
            with LoggingContext("S1510 - UDF Database Connection"):
                logger.info(f"Creating new SQLite connection to {db_path}...")
                try:
                    conn, cursor = sqlite_db_open(db_path)
                except Exception as e:
                    logger.error(f"Failed to create SQLite connection: {str(e)} Check DB path: {db_path}")
                    raise
                close_connection = True
            logger.key("S1510 - UDF Database connection established")

            # Validate WHERE clause
            with LoggingContext("S1520 - UDF WHERE clause Validation"):
                logger.debug(f"WHERE clause: {where_clause}")
                if not where_clause or not where_clause.strip():
                    logger.error("WHERE clause is empty or invalid")
                    raise ValueError("WHERE clause is empty or invalid")
                logger.info(f"WHERE clause validated: {where_clause}")
            logger.key(f"S1520 - OK: WHERE clause validated: {where_clause}")
            
            try:
                with LoggingContext("S1530 - Read UDF as JSON"):
                    udf_json_data = read_udf_as_json(cursor, where_clause)
                logger.key("S1530 - OK: UDF to JSON Read completed successfully exported")
            except Exception as e:
                logger.error(f"S1530 - ERROR: Failed to read UDF data: {str(e)}")
                raise

            # Write JSON to file
            try:
                with LoggingContext("S1540 - Write UDF to JSON"):
                    output_json_file = write_udf_json_to_file(udf_json_data, where_clause, write_json_dir)
                logger.key(f"S1540 - OK: UDF to JSON Write completed successfully - {output_json_file}")
            except Exception as e:
                logger.error(f"S1540 - ERROR: Failed to write JSON file: {str(e)}")
                raise
            
        logger.key(f"UDF to JSON Export completed successfully. Find JSON file in {write_json_dir}")
        return 0
    except Exception as e:
        logger.error(f"ERROR: UDF to JSON Export failed: {str(e)}")
        return 1
    finally:
        # Close connection if we opened it
        if close_connection and conn:
            with LoggingContext("S1550 - UDF Database Connection"):
                sqlite_db_close(conn)
                logger.debug("SQLite connection closed")

if __name__ == "__main__":
    sys.exit(main())
