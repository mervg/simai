import sqlite3
from logger.logger import logger, LoggingContext
from typing import Tuple

def sqlite_db_open(db_path: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Set up SQLite database with required tables.

    Creates or updates the SQLite database with tables for storing UDF records.

    Args:
        db_path (str): Path to the SQLite database file

    Returns:
        tuple: A tuple containing the database connection and cursor objects
    """
    try:
        logger.info(f"Setting up SQLite database {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create UDF table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS UDF (
                udf_id INTEGER PRIMARY KEY AUTOINCREMENT,
                udf_createddate DATETIME DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'UTC')),
                udf_createdby TEXT,
                udf_lastmodifieddate DATETIME DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'UTC')),
                udf_lastmodifiedby TEXT,
                udf_source_dm TEXT,
                udf_type TEXT NOT NULL,
                udf_entity_id TEXT UNIQUE NOT NULL,
                udf_date TEXT,
                udf_name TEXT,
                store TEXT,
                banner TEXT,
                territory TEXT,
                `store segment` TEXT,
                vendor TEXT,
                brand TEXT,
                `product group` TEXT,
                product TEXT,
                `product category` TEXT,
                `product subcategory` TEXT,
                user TEXT,
                `user type` TEXT,
                meta_dates TEXT,
                meta_product TEXT,
                meta_store TEXT,
                meta_user TEXT,
                meta_tags TEXT,
                meta_context TEXT,
                meta_result TEXT
            )
        """)
        
        # Create lastmodifieddate trigger for UDF
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS UDF_lastmodifieddate
            AFTER UPDATE ON UDF
            BEGIN
                UPDATE UDF
                SET udf_lastmodifieddate = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', 'UTC')
                WHERE udf_id = NEW.udf_id;
            END;
        """)
        
        logger.info("SQLite Database setup complete.")
        return conn, cursor
    except sqlite3.Error as e:
        logger.error(f"SQLite error occurred: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when setting up SQLite database: {str(e)}")
        raise

def sqlite_db_udf_write(conn: sqlite3.Connection, cursor: sqlite3.Cursor, udf_record: dict):
    """Write a UDF record to the database. If the record already exists based on the `udf_entity_id` field, it will be updated. Otherwise, a new record will be inserted.
    
    Args:
        conn (sqlite3.Connection): Database connection object
        cursor (sqlite3.Cursor): Database cursor object
        udf_record (dict): Dictionary containing UDF record data
        
    Raises:
        sqlite3.Error: If a database error occurs during the operation
        KeyError: If the required 'udf_entity_id' key is missing from the udf_record
    """
    try:
        # Basic input validation
        if 'udf_entity_id' not in udf_record:
            logger.error("Missing required 'udf_entity_id' field in UDF record")
            raise KeyError("Missing required 'udf_entity_id' field in UDF record")
            
        # Check if record exists
        cursor.execute("SELECT udf_id FROM UDF WHERE udf_entity_id = ?", (udf_record['udf_entity_id'],))
        existing_record = cursor.fetchone()
        
        if existing_record:
            # Update existing record
            set_clauses = ', '.join([f"`{col}` = ?" for col in udf_record.keys() if col != 'udf_id'])
            values = [udf_record[col] for col in udf_record.keys() if col != 'udf_id']
            values.append(udf_record['udf_entity_id'])
            
            update_sql = f"UPDATE UDF SET {set_clauses} WHERE udf_entity_id = ?"
            cursor.execute(update_sql, values)
            logger.debug(f"Updated existing UDF record with entity_id: {udf_record['udf_entity_id']}")
        else:
            # Insert new record
            columns = ', '.join([f'`{col}`' for col in udf_record.keys()])
            placeholders = ', '.join(['?'] * len(udf_record))
            insert_sql = f"INSERT INTO UDF ({columns}) VALUES ({placeholders})"
            cursor.execute(insert_sql, list(udf_record.values()))
            logger.debug(f"Inserted new UDF record with entity_id: {udf_record['udf_entity_id']}")
            
    except sqlite3.Error as e:
        logger.error(f"Database error occurred while writing UDF record: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while writing UDF record: {str(e)}")
        raise
    
    # Removed conn.commit() to allow batch processing

def sqlite_db_close(conn: sqlite3.Connection) -> None:
    """Close the SQLite database connection.
    
    Args:
        conn (sqlite3.Connection): Database connection object
    """
    try:
        logger.info("Closing SQLite database connection...")
        conn.commit()
        conn.close()
        logger.info("SQLite database connection closed successfully")
    except sqlite3.Error as e:
        logger.error(f"SQLite error when closing connection: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when closing SQLite connection: {str(e)}")
        raise
