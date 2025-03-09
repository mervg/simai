"""
Export module for WebSIMAI application.

This module contains functions for converting DataFrame data to
structured JSON format and generating JSON files for export.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st
from loguru import logger

from logger.logger import LoggingContext


def df_to_structured_json(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to structured JSON format.
    
    This function adapts the logic from udf_to_json.read_udf_as_json to work 
    with a pandas DataFrame instead of directly querying SQLite.
    
    Args:
        df: DataFrame containing UDF records
        
    Returns:
        Structured JSON object with UDF data
    """
    with LoggingContext("S0330 - DataFrame to JSON Conversion"):
        try:
            # Initialize the result dictionary
            result = {
                "metadata": {
                    "record_count": len(df),
                    "export_time": datetime.now().isoformat(),
                    "filters_applied": list(st.session_state.filters.keys())
                },
                "records": []
            }
            
            # Process each row in the DataFrame
            for _, row in df.iterrows():
                # Convert row to dictionary
                record = row.to_dict()
                
                # Add to records list
                result["records"].append(record)
            
            logger.key("S0330 - OK: DataFrame converted to JSON successfully")
            return result
            
        except Exception as e:
            logger.error(f"S0330 - FAILED: Error converting DataFrame to JSON: {str(e)}")
            raise ValueError(f"Failed to convert data to JSON: {str(e)}")


def generate_json_file(json_data: Dict[str, Any], filename: str) -> str:
    """
    Generate JSON file and return download path.
    
    This function adapts the logic from udf_to_json.write_udf_json_to_file
    
    Args:
        json_data: Structured JSON data to write
        filename: Name of the output file
        
    Returns:
        Path to the generated file
    """
    with LoggingContext("S0330 - JSON File Generation"):
        try:
            # Ensure exports directory exists
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate file path
            file_path = os.path.join(export_dir, filename)
            
            # Write JSON to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON file generated at {file_path}")
            logger.key("S0330 - OK: JSON file generated successfully")
            
            return file_path
            
        except Exception as e:
            logger.error(f"S0330 - FAILED: Error generating JSON file: {str(e)}")
            raise IOError(f"Failed to write JSON file: {str(e)}")
