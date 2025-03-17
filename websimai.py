import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import os
import sqlite3
import json
import tiktoken
from typing import Dict, List, Tuple, Any, Optional, Union
import google.generativeai as genai
import datetime
import markdown2
from xhtml2pdf import pisa
import io

# Custom JSON encoder for handling pandas Timestamp objects
class TimestampJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles pandas Timestamp objects.
    Converts Timestamp objects to ISO format strings for JSON serialization.
    """
    def default(self, obj):
        if pd.api.types.is_datetime64_any_dtype(obj) or isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

SQLITE_DB_PATH = os.path.dirname(__file__)
SQLITE_DB_FILENAME = os.path.join(SQLITE_DB_PATH, 'UDF.db')

# Pricing constants for Gemini models (per million tokens)
GEMINI_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input_char": 0.00001875, "output_char": 0.000075},
    "gemini-1.5-pro": {"input_char": 0.0003125, "output_char": 0.00125}
}

def st_init_page() -> bool:
    st.set_page_config(
        page_title="WebSIMAI - UDF Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    return True

def load_udf_to_df(db_path: str) -> pd.DataFrame:
    df = None
    print("load_udf_to_df > Loading UDF data")
    try:
        # Get database connection
        try:
            # Check if database file exists
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file not found at {db_path}")
            
            # Establish connection
            print("load_udf_to_df > Establishing connection")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test connection
            cursor.execute("SELECT sqlite_version();")
            cursor.fetchone()
            
        except sqlite3.Error as e:
            err_msg = f"Failed to connect to database: {str(e)}"
            st.error(err_msg)
            raise sqlite3.Error(err_msg)


        # Load UDF data into DataFrame
        try:
            # Execute query to get all UDF data
            query = """
            SELECT * FROM UDF
            """
            
            # Load data from UDF and Store in session state
            print("load_udf_to_df > Loading data to DataFrame")
            df = pd.read_sql_query(query, conn)

        except Exception as e:
            # Create empty DataFrame as fallback
            df = pd.DataFrame()
            st.error(f"Failed to load UDF data: {str(e)}")
        
        # Close database connection
        print("load_udf_to_df > Closing connection")
        conn.close()
        
        print("load_udf_to_df > All Good: Returning DataFrame")
        return df

    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        # Return empty DataFrame
        return pd.DataFrame()

def convert_date_format_agnostic(date_str):
    """
    Attempt to convert a date string to datetime using multiple formats.
    If conversion fails, return the original string.
    
    Args:
        date_str: The date string to convert
        
    Returns:
        str: The date in ISO format if conversion was successful, or the original string if not
    """
    if pd.isna(date_str):
        return date_str
        
    # Try multiple date formats
    formats_to_try = [
        'ISO8601',      # ISO format with T separator
        '%Y-%m-%dT%H:%M:%S',  # Explicit ISO format
        '%Y-%m-%d %H:%M:%S',  # Standard datetime format
        '%Y-%m-%d'      # Just date
    ]
    
    for fmt in formats_to_try:
        try:
            # Convert to datetime and then to ISO format string
            dt = pd.to_datetime(date_str, format=fmt)
            return dt.isoformat()
        except Exception:
            continue
            
    # If all conversions fail, keep the original value and log a warning
    print(f"Warning: Failed to convert date value '{date_str}' to datetime")
    return date_str

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters", value=True)

    if not modify:
        return df

    df = df.copy()

    # Only convert the udf_date column to datetime using a format-agnostic approach 
    if 'udf_date' in df.columns and is_object_dtype(df['udf_date']):
        # Define a function to convert a single date value with multiple format attempts
        # Apply the conversion function to the udf_date column
        df['udf_date'] = df['udf_date'].apply(convert_date_format_agnostic)
        
        # Remove timezone information if present
        if is_datetime64_any_dtype(df['udf_date']):
            df['udf_date'] = df['udf_date'].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if isinstance(df[column], pd.Categorical) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    format="DD/MM/YYYY",
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input, case=False)]

    return df

def convert_selected_df_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    """
    This function processes the DataFrame records into a standardized JSON structure, and identifies relationships 
    between different UDF record types to create either a hierarchical or flat structure.
    
    Args:
        df (pd.DataFrame): DataFrame containing selected UDF records
        
    Returns:
        Dict[str, Any]: Structured JSON data with appropriate relationships between records
    """
    # Check if DataFrame is empty
    if df.empty:
        return {"records": []}
        
    # Get column names from DataFrame
    columns = df.columns.tolist()
    
    # Convert DataFrame rows to list of dictionaries with native Python types
    records = df.to_dict('records')
    
    record_count = len(records)
        
    if record_count == 0:
        return {"records": []}
        
    # Process records into structured format
    structured_data, type_groups = process_json_structure(records, columns)

    print(f"Type groups keys: {type_groups.keys()}")
    print(f"Number of tasksdm_task records: {len(type_groups.get('tasksdm_task', []))}")
    print(f"Number of tasksdm_task_instance records: {len(type_groups.get('tasksdm_task_instance', []))}")

    # Discover relationships between types
    relationships = discover_relationships(type_groups)

    print(f"Relationships found: {relationships}")
    if 'tasksdm_task' in type_groups:
        print(f"Sample task entity_id: {type_groups['tasksdm_task'][0].get('udf_entity_id', 'none')}")
    if 'tasksdm_task_instance' in type_groups:
        print(f"Sample instance entity_id: {type_groups['tasksdm_task_instance'][0].get('udf_entity_id', 'none')}")

    # If we have task and task_instance types with relationships, create a nested structure
    if 'tasksdm_task' in type_groups and 'tasksdm_task_instance' in type_groups and relationships:
        print("OK: Building task hierarchy")
        structured_data = build_task_hierarchy(type_groups)
    else:
        print("OK: Using flat structure")
        structured_data = {"records": [record for record in structured_data["records"]]}
    
    return structured_data

def process_json_structure(records: List[Dict[str, Any]], columns: List[str]) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Analyzes UDF records and determines appropriate JSON structure.
    
    Processes records into structured dictionaries by:
    1. Converting records into dictionaries with proper JSON parsing
    2. Grouping records by udf_type for relationship analysis
    3. Preparing data for hierarchical or flat structure creation
    
    Args:
        records (List[Dict[str, Any]]): Records as dictionaries
        columns (List[str]): Column names for the records
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]: 
            - Structured JSON data dictionary with 'records' key
            - Records grouped by udf_type for relationship analysis
    """
    # Convert records to dictionaries with JSON parsing for meta fields 
    record_dicts = []
    for record in records:
        record_dict = {}
        for col in columns:
            # Skip NULL values
            if record[col] is None:
                continue
                
            # Parse JSON fields
            if col in ["meta_dates", "meta_product", "meta_store", "meta_user", 
                      "meta_tags", "meta_context", "meta_result"]:
                try:
                    record_dict[col] = json.loads(record[col]) if isinstance(record[col], str) else record[col]
                except json.JSONDecodeError:
                    record_dict[col] = record[col]
            else:
                record_dict[col] = record[col]
        record_dicts.append(record_dict)
    
    # Group by udf_type
    type_groups = {}
    for record in record_dicts:
        udf_type = record.get('udf_type')
        if udf_type not in type_groups:
            type_groups[udf_type] = []
        type_groups[udf_type].append(record)
    
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
        return relationships
    
    # Check each pair of types for potential relationships
    for child_type, child_records in type_groups.items():
        for parent_type, parent_records in type_groups.items():
            if child_type == parent_type:
                continue
                
            # Check if child entity_ids contain parent entity_ids
            for child_record in child_records:
                child_entity_id = child_record.get('udf_entity_id', '')
                
                for parent_record in parent_records:
                    parent_entity_id = parent_record.get('udf_entity_id', '')
                    
                    # If child entity_id starts with parent entity_id and has additional parts
                    if (child_entity_id.startswith(parent_entity_id) and 
                        child_entity_id != parent_entity_id and
                        len(child_entity_id.split('-')) > len(parent_entity_id.split('-'))):
                        relationships[child_type] = parent_type
                        break
                
                if child_type in relationships:
                    break
    
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
    print("Building task hierarchy...")
    # Create a structure with tasks and their instances
    tasks = type_groups.get('tasksdm_task', [])
    instances = type_groups.get('tasksdm_task_instance', [])
    
    result = {
        "tasks": []
    }
    
    # Process each task
    for task in tasks:
        print(f"Processing task: {task['udf_entity_id']}")
        task_entity_id = task.get('udf_entity_id', '')
        task_with_instances = {
            "task": task,
            "instances": []
        }
        
        # Find instances for this task
        for instance in instances:
            print(f"Checking instance: {instance['udf_entity_id']}")
            instance_entity_id = instance.get('udf_entity_id', '')
            # Check if instance belongs to this task
            if instance_entity_id.startswith(task_entity_id + '-'):
                print(f"OK: Instance {instance_entity_id} belongs to task {task_entity_id}")
                task_with_instances["instances"].append(instance)
        result["tasks"].append(task_with_instances)
    
    return result

def generate_json_of_selected() -> Dict[str, Any]:
    print("Generating JSON...")
    st.spinner("Generating JSON...")
    st.session_state.selected_in_json = convert_selected_df_to_json(st.session_state.selected_df)
    st.toast("JSON generated successfully")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    json_filename = f"JSON_{timestamp}.json"
    st.session_state.json_filename = json_filename

    return st.session_state.selected_in_json

def add_instances(toggle: bool) -> None:
    """
    Add task instances to selected_df for each selected task.
    
    Args:
        toggle (bool): If True, add instances; if False, do nothing
    """
    if not toggle:
        return
        
    # Get tasks from selected_df
    tasks = st.session_state.selected_df[
        st.session_state.selected_df['udf_type'] == 'tasksdm_task'
    ]
    
    if tasks.empty:
        return
        
    # Get all instances from filtered_df
    instances = st.session_state.df[
        st.session_state.df['udf_type'] == 'tasksdm_task_instance'
    ]
    
    if instances.empty:
        return
    
    # Find and add instances for each task
    instances_to_add = []
    for _, task in tasks.iterrows():
        task_id = task['udf_entity_id']
        matching_instances = instances[
            instances['udf_entity_id'].str.startswith(task_id + '-')
        ]
        instances_to_add.append(matching_instances)
    
    if instances_to_add:
        # Combine all instances and add to selected_df
        all_instances = pd.concat(instances_to_add)
        st.session_state.selected_df = pd.concat(
            [st.session_state.selected_df, all_instances]
        ).drop_duplicates()

def selection_changed():
    st.session_state.selected_in_json = {}
    reset_report_generation()

def count_tokens(
    data: Union[str, Dict, List, pd.DataFrame, Any], 
    model: Optional[str] = "cl100k_base",
    truncate_to: Optional[int] = None) -> int:
    """
    Count tokens for various data types using tiktoken.
    
    This function serves as a universal wrapper for tiktoken's token counting,
    supporting different data types including strings, dictionaries, lists,
    pandas DataFrames, and other objects.
    
    Args:
        data: The data to count tokens for. Can be a string, dictionary, list, 
              pandas DataFrame, or other object.
        model: The name of the tiktoken model to use for counting.
        truncate_to: Optional maximum number of characters to consider for token counting.
                     If provided, the text will be truncated to this length before counting.
    
    Returns:
        int: The number of tokens in the data.
    """
    # Get the appropriate encoder for the model
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding(model)
    
    # Convert data to text based on its type
    if isinstance(data, str):
        text = data
    elif isinstance(data, (dict, list)):
        # Convert dict/list to JSON string
        text = json.dumps(data, separators=(',', ":"), cls=TimestampJSONEncoder)
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to dict records and then to JSON
        records = data.to_dict('records')
        text = json.dumps(records, separators=(',', ":"), cls=TimestampJSONEncoder)
    
    else:
        # Fall back to string representation for other types
        text = str(data)
    
    # Optionally truncate text for estimation on large data
    if truncate_to and len(text) > truncate_to:
        # Take a sample from the beginning, middle and end for better estimation
        third = truncate_to // 3
        text = text[:third] + text[len(text)//2-third//2:len(text)//2+third//2] + text[-third:]
    
    # Count tokens
    return len(encoder.encode(text))

def calculate_all_tokens(
    prompt: Optional[str] = None, 
    json_data: Optional[Union[Dict, List, pd.DataFrame]] = None,
    response: Optional[str] = None,
    model: Optional[str] = "cl100k_base",
    gemini_model: Optional[str] = "gemini-2.0-flash") -> Dict[str, int]:
    """
    Calculate token counts and costs for prompt, JSON data, and response.
    
    Args:
        prompt: Optional prompt text
        json_data: Optional JSON data (dict, list, or DataFrame)
        response: Optional response text from LLM
        model: Encoding model to use for token counting
        gemini_model: Gemini model name for cost calculation
    
    Returns:
        Dictionary containing token counts and costs for each component and totals
    """
    results = {
        "prompt_tokens": count_tokens(prompt or "", model),
        "json_tokens": count_tokens(json_data or {}, model),
        "output_tokens": count_tokens(response or "", model)
    }
    
    results["total_input_tokens"] = results["prompt_tokens"] + results["json_tokens"]
    results["total_tokens"] = results["total_input_tokens"] + results["output_tokens"]
    
    # Calculate costs based on token counts
    costs = calculate_gemini_cost(gemini_model, results["total_input_tokens"], results["output_tokens"])
    results["input_cost"] = costs["input_cost"]
    results["output_cost"] = costs["output_cost"]
    results["total_cost"] = costs["total_cost"]
    
    return results

def calculate_gemini_cost(model_name, input_tokens, output_tokens):
    """
    Calculate cost based on token usage for Gemini models.
    
    Args:
        model_name: The Gemini model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Dictionary with input_cost, output_cost, and total_cost in USD
    """
    # Default to gemini-2.0-flash if model not found
    model_name = model_name if model_name in GEMINI_PRICING else "gemini-2.0-flash"
    pricing = GEMINI_PRICING[model_name]
    
    # For token-based models (Gemini 2.0)
    if "input" in pricing:
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
    # For character-based models (Gemini 1.5)
    else:
        # Convert tokens to approximate characters (4 chars per token)
        input_chars = input_tokens * 4
        output_chars = output_tokens * 4
        
        input_cost = (input_chars / 1_000) * pricing["input_char"]
        output_cost = (output_chars / 1_000) * pricing["output_char"]
    
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def llm_connect(api_key: str, model_name: str = "gemini-2.0-flash"):
    """
    Connects to the Gemini API.

    Args:
        api_key (str): Gemini API key.
        model_name (str): Name of the Gemini model to use.

    Returns:
        Any: Initialized Gemini model object, or None if connection fails.
    """
    try:
        genai.configure(api_key=api_key) # Configure API key globally once
        model = genai.GenerativeModel(model_name)
        print(f"Successfully connected to Gemini model: {model_name}")
        return model
    except Exception as e:
        st.error(f"Failed to connect to Gemini API: {e}")
        return None

def prepare_prompt(
    system_prompt: str, 
    user_prompt: str, 
    json_data: Optional[Union[Dict, List]] = None
    ) -> str:
    """
    Prepare a prompt for the LLM by combining system prompt, user prompt, and optional JSON data.
    
    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        json_data: Optional JSON data to include in the prompt
        
    Returns:
        str: The combined prompt
    """
    prompt_content = f"{system_prompt}\n\n---\n\nUser Query: {user_prompt}"
    if json_data:
        # prompt_content += f"\n\n---\n\nData (JSON):\n```json\n{json.dumps(json_data, indent=2)}\n```"
        prompt_content += f"\n\n---\n\nData (JSON):\n```json\n{json.dumps(json_data, separators=(',', ":"), cls=TimestampJSONEncoder)}\n```" ## no indentation added for AI JSON data, otherwise the JSON becomes too bloated

    # Store system and user prompts in session state for report generation log
    st.session_state.report_log_sys_prompt = system_prompt
    st.session_state.report_log_user_prompt = user_prompt
    
    return prompt_content

def generate_llm_response(model: Any, system_prompt: str, user_prompt: str, json_data: Optional[Dict] = None, stream: bool = False) -> Union[str, Any]:
    """
    Generates a response from the Gemini LLM based on prompts and optional JSON data.
    Supports both streaming and non-streaming responses.
    For non-streaming responses, uses Gemini's native token counting for accurate token statistics.

    Args:
        model (Any): Initialized Gemini model object from llm_connect.
        system_prompt (str): System-level instructions for the LLM.
        user_prompt (str): User's query or request.
        json_data (Optional[Dict], optional): JSON data to provide context to the LLM. Defaults to None.
        stream (bool): If True, returns a streaming response; otherwise, returns the full response text.

    Returns:
        Union[str, Any]: Text response from the LLM if stream=False, or a streaming response iterator if stream=True.
    """
    prompt_content = prepare_prompt(system_prompt, user_prompt, json_data)
    # Store system and user prompts in session state for report generation log
    st.session_state.report_log_sys_prompt = system_prompt
    st.session_state.report_log_user_prompt = user_prompt
    
    # Pre-calculate estimated token counts (used as fallback if native counting fails)
    estimated_tokens = calculate_all_tokens(
        prompt=prompt_content,
        json_data=json_data,
        response=None  # No response yet
    )

    try:
        if stream:
            response = model.generate_content(prompt_content, stream=True)
            # For streaming, we use the estimated token counts
            st.session_state.report_token_stats = estimated_tokens
            return response  # Return the response object directly for streaming
        else:
            # Create generation config with temperature parameter
            generation_config = {
                "temperature": 0.3
            }
            response = model.generate_content(prompt_content, generation_config=generation_config)
            
            if response.parts:  # Check if response is valid
                report_text = response.text
                st.session_state.generated_report = report_text
                
                # Get accurate token counts from Gemini's response metadata if available
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    # Extract actual token counts from response metadata
                    actual_tokens = {
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', estimated_tokens["prompt_tokens"]),
                        "json_tokens": estimated_tokens["json_tokens"],  # Keep estimated JSON tokens since Gemini doesn't separate this
                        "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', estimated_tokens["output_tokens"]),
                        "total_input_tokens": getattr(response.usage_metadata, 'prompt_token_count', estimated_tokens["total_input_tokens"]),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', estimated_tokens["total_tokens"])
                    }
                    st.session_state.report_token_stats = actual_tokens
                else:
                    # Fall back to estimated counts if metadata not available
                    output_tokens = count_tokens(report_text)
                    estimated_tokens["output_tokens"] = output_tokens
                    estimated_tokens["total_tokens"] = estimated_tokens["total_input_tokens"] + output_tokens
                    st.session_state.report_token_stats = estimated_tokens
                
                return report_text
            else:
                return "No response from AI. Please check your prompt and data."
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        # In case of error, still provide estimated token counts
        st.session_state.report_token_stats = estimated_tokens
        return f"Error: {e}"

def reset_report_generation():
    st.session_state.report_log_sys_prompt = ""
    st.session_state.report_log_user_prompt = ""
    st.session_state.report_json_uploaded = None
    st.session_state.generated_report = ""
    st.session_state.gen_disabled = True
    st.session_state.is_report_ready = False
    st.session_state.report_generating = False
    st.session_state.report_generated = False
    st.session_state.json_source_option = "Upload JSON File"
    st.session_state.report_token_stats = {}
    
    # Use a special flag to handle file uploader reset
    # This avoids direct manipulation of the file uploader widget state
    st.session_state.reset_report_widgets = True
    
    # Keep the json_source_option as is to avoid recreating the file uploader
    # with the same key, which would cause an error
    
    #st.toast("Report generation reset successfully!")

def reset_chat():
    """
    Reset the chat history and related session state variables.
    Clears all messages and resets token counters to their initial state.
    Also reinitializes the Gemini chat session.
    """
    # Reset message history and token counters
    st.session_state.messages = []
    st.session_state.prompt_tokens = 0
    st.session_state.file_tokens = 0
    st.session_state.total_tokens_out = 0
    st.session_state.total_tokens_in = 0
    st.session_state.chat_token_stats = {
        'total_tokens': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'prompt_tokens': 0,
        'json_tokens': 0,
        'response_tokens': 0,
        'using_native_counts': False,
        'input_cost': 0.0,
        'output_cost': 0.0,
        'total_cost': 0.0
    }
    st.session_state.chat_generating = False
    
    # Reinitialize the Gemini chat session
    if "gemini_chat_session" in st.session_state:
        # Get the existing model connection
        gemini_model = llm_connect(st.secrets["GEMINI_API_KEY"]) # Connect to model only once
        if gemini_model:
            # Create a fresh chat session
            st.session_state.gemini_chat_session = gemini_model.start_chat()
            st.toast("Chat history reset successfully!")
        else:
            st.error("Failed to reinitialize Gemini Chat model.")
            st.stop() # Stop if model initialization fails

def format_chat_history_as_markdown() -> str:
    """
    Format the current chat history as a Markdown document.
    
    Returns:
        str: Chat history formatted as Markdown
    """
    if not st.session_state.messages:
        return "# Chat History\n\nNo messages in chat history."
    
    markdown_content = "# Chat History\n\n"
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            markdown_content += f"## User Input:\n\n{message['prompt']}\n\n"
            if "file_tokens" in message and message["file_tokens"] > 0 and message["file_name"] is not None:
                markdown_content += f"*File: \"{message['file_name']}\" (tokens: {message['file_tokens']:,})*\n\n"
        elif message["role"] == "AI":
            markdown_content += f"## Response:\n\n{message['response']}\n\n"
            markdown_content += "---\n\n"
    
    return markdown_content

def convert_markdown_to_pdf_python(markdown_content: str, output_filename: str) -> bytes:
    """
    Converts Markdown content to PDF using markdown2 and xhtml2pdf (Python libraries).
    Optimized for better table rendering and overall PDF quality.

    Args:
        markdown_content (str): The Markdown text to convert.
        output_filename (str): The desired filename for the PDF.
        
    Returns:
        bytes: The PDF file content as bytes, ready for download.
              Returns None and prints an error if conversion fails.
    """
    try:
        # Define CSS that works well with xhtml2pdf's limitations
        css_styles = """
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.3;
            margin: 1cm;
        }
        
        h1, h2, h3, h4, h5, h6 {
            margin-top: 0.5cm;
            margin-bottom: 0.2cm;
            color: #2c3e50;
            page-break-after: avoid;
        }
        
        h1 { font-size: 18pt; }
        h2 { font-size: 16pt; }
        h3 { font-size: 14pt; }
        h4 { font-size: 12pt; }
        
        p {
            margin-bottom: 0.2cm;
        }
        
        /* Table styling optimized for xhtml2pdf */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.5cm 0;
            -pdf-keep-with-next: false;
            -pdf-keep-in-frame-mode: shrink;
        }
        
        th {
            font-weight: bold;
            background-color: #f2f2f2;
            border: 1px solid #dddddd;
            padding: 4px 8px;
            text-align: left;
            vertical-align: middle;
        }
        
        td {
            border: 1px solid #dddddd;
            padding: 4px 8px;
            text-align: left;
            vertical-align: top;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        /* Improved list styling for better spacing */
        ul, ol {
            margin-top: 0.1cm;
            margin-bottom: 0.1cm;
            margin-left: 0.3cm;
            padding-left: 0.3cm;
        }
        
        li {
            margin-top: 0;
            margin-bottom: 0.05cm;
            line-height: 1.1;
            padding-left: 0;
        }
        
        /* Nested lists */
        li ul, li ol {
            margin-top: 0.05cm;
            margin-bottom: 0;
        }
        
        /* Bullet points appearance */
        ul {
            list-style-type: disc;
        }
        
        ul ul {
            list-style-type: circle;
        }
        
        ul ul ul {
            list-style-type: square;
        }
        
        /* Code blocks */
        pre, code {
            font-family: Courier, monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            padding: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        /* Images */
        img {
            max-width: 100%;
            height: auto;
        }
        """
        
        # Convert Markdown to HTML with extended extras for better table support 
        html_content = markdown2.markdown(
            markdown_content,
            extras=[
                'tables',               # Enable table support
                'code-friendly',        # Better code block handling
                'cuddled-lists',        # Better list rendering
                'markdown-in-html',     # Allow markdown inside HTML blocks
                'break-on-newline',     # Line breaks on newlines
                'header-ids'            # Add IDs to headers
            ]
        )
        
        # Create a complete HTML document with proper DOCTYPE and meta tags
        full_html = f"""<!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>{output_filename}</title>
                            <style type="text/css">
                        {css_styles}
                            </style>
                        </head>
                        <body>
                        {html_content}
                        </body>
                        </html>"""
        
        # Create a BytesIO buffer for the PDF output
        pdf_buffer = io.BytesIO()
        
        # Convert HTML to PDF with optimized settings
        pdf_status = pisa.CreatePDF(
            src=full_html,              # Complete HTML document
            dest=pdf_buffer,            # Output buffer
            encoding='UTF-8',           # Ensure proper character encoding
            raise_exception=False,      # Don't raise exceptions for warnings
            xhtml=False                 # Set to False to avoid XHTMLParser error 
        )
        
        if pdf_status.err:
            st.error(f"Error converting HTML to PDF using xhtml2pdf: {pdf_status.err}")
            return None
            
        # Get the PDF content from the buffer
        pdf_buffer.seek(0)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"Error during Python-based PDF conversion: {e}")
        return None

def convert_markdown_to_pdf_opt2(markdown_content: str, output_filename: str) -> bytes:
    pass

def get_default_system_prompt():
    """
    Read the default system prompt from a file and prepend today's date.
    
    Returns:
        str: The content of the default_sys_prompt.md file with today's date prepended,
             or an empty string if the file doesn't exist.
    """
    try:
        # Get today's date in the required format
        today = datetime.datetime.now().strftime("%d %B %Y")
        date_prefix = f"Today's date: {today}\n\n"
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(script_dir, "default_sys_prompt.md")
        
        # Check if the file exists
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                return date_prefix + file_content
        else:
            print(f"Warning: Default system prompt file not found at {prompt_file_path}")
            return date_prefix
    except Exception as e:
        print(f"Error reading default system prompt: {e}")
        return ""

st_init_page() #called outside of main() to not  be run once and  not be incl uded into reruns 

def main():
    print("We are in main, RERUN")
    # Load UDF data into DataFrame
    if "df" not in st.session_state:
        st.spinner("Initializing: Loading UDF data...")        
        st.session_state.df = load_udf_to_df(SQLITE_DB_FILENAME)
        st.toast("UDF data loaded successfully")

    if "df_total_recs" not in st.session_state:
        st.session_state.df_total_recs = len(st.session_state.df)

    # Display interactive dataframe with multi-row selection
    st.write(f"UDF Loaded, Total Records: **{st.session_state.df_total_recs}**")
    st.session_state.filtered_df = filter_dataframe(st.session_state.df)
    st.write(f"Filtered down to: **{len(st.session_state.filtered_df)}**")

    st.session_state.selected_rows = st.dataframe(st.session_state.filtered_df, 
                                    key='dataframe',
                                    hide_index = True,
                                    selection_mode="multi-row",
                                    on_select=selection_changed,
                                    use_container_width=True)

    st.write("Select rows in the dataset above to proceed...")
    # Filter based on selection
    if "selected_df" not in st.session_state:
        st.session_state.selected_df = pd.DataFrame() #st.session_state.df.copy()
    else:
        selected_indices = st.session_state.selected_rows["selection"]["rows"]
        st.session_state.selected_df = st.session_state.filtered_df.iloc[selected_indices].copy()


    col11, col12 = st.columns(2)

    with col12:
        add_instances_toggle = st.checkbox("Add Task Instances to selected Tasks (from original dataset, if needed)", on_change=selection_changed)
        add_instances(add_instances_toggle)

    with col11:
        st.session_state.selected_df_recs = len(st.session_state.selected_df)
        st.session_state.selected_df_tokens = count_tokens(st.session_state.selected_df)
        st.write(f"\nSelected records: **{st.session_state.selected_df_recs}**, Tokens (est): **{st.session_state.selected_df_tokens:,}**")

    st.dataframe(st.session_state.selected_df, hide_index=True, use_container_width=True)

    # JSON generation
    if "selected_in_json" not in st.session_state:
        st.session_state.selected_in_json = {}

    json_col1, json_col2, json_col3, json_col4 = st.columns([2,3,5,2])

    if st.session_state.selected_df_recs >0: #and len(st.session_state.selected_in_json) == 0:
        json_col1.button("Generate JSON", on_click=generate_json_of_selected)

    if len(st.session_state.selected_in_json) > 0:
        with json_col2:
            JSON_preview_msg = "Preview JSON"
            with st.popover(JSON_preview_msg):
                st.json(st.session_state.selected_in_json, expanded=3)

    if "json_filename" not in st.session_state:
        st.session_state.json_filename = "selected.json"

    if st.session_state.selected_in_json:
        with json_col3:
            st.session_state.json_filename = st.text_input("JSON filename", value=st.session_state.json_filename, label_visibility="collapsed")
        with json_col4:
            # st.download_button("Download JSON", json.dumps(st.session_state.selected_in_json, indent=2), st.session_state.json_filename)
            st.download_button("Download JSON", json.dumps(st.session_state.selected_in_json, separators=(',', ":"), cls=TimestampJSONEncoder), st.session_state.json_filename) ## no indentation added for AI JSON data, otherwise the JSON becomes too bloated

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.prompt_tokens = 0
        st.session_state.file_tokens = 0
        st.session_state.total_tokens_out = 0
        st.session_state.total_tokens_in = 0
        st.session_state.chat_generating = False # Add chat_generating state
        
    # Ensure chat_token_stats is always initialized separately 
    if "chat_token_stats" not in st.session_state:
        st.session_state.chat_token_stats = {
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'prompt_tokens': 0,
            'json_tokens': 0,
            'response_tokens': 0,
            'using_native_counts': False,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

    #if st.session_state.selected_in_json: # show chat bot container only if JSON is generated
    gen_tab, chat_tab = st.tabs(["Generate Report", "Chat with AI"])
    with chat_tab:
        # Initialize ChatSession if it doesn't exist in session_state
        if "gemini_chat_session" not in st.session_state:
            gemini_model = llm_connect(st.secrets["GEMINI_API_KEY"]) # Connect to model only once
            if gemini_model:
                st.session_state.gemini_chat_session = gemini_model.start_chat()
            else:
                st.error("Failed to initialize Gemini Chat model.")
                st.stop() # Stop if model initialization fails

        chat_session = st.session_state.gemini_chat_session # Get the ChatSession from session state

        # "If you finished preparing your JSON file and downloaded it, proceed to chat - input your fancy prompt, attach JSON file, and let it do its magic"
        chat_placeholder = st.container(height=700) # Changed to placeholder

        with chat_placeholder: # Use placeholder
            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message(message["role"]):
                        st.markdown(f"{message["prompt"]}")
                        if "file_tokens" in message and message["file_tokens"] > 0 and message["file_name"] is not None:
                            st.markdown(f"File: \"*{message["file_name"]}*\" (tokens: {message["file_tokens"]:,})")
                elif message["role"] == "AI":
                    with st.chat_message(message["role"]):
                        st.markdown(message["response"]) # No changes here, response is already rendered

        prompt = st.chat_input("Prompt, please?", accept_file=True, file_type="json", disabled=st.session_state.chat_generating) # Disable input while generating

        # React to user input
        with chat_placeholder: # Use placeholder
            if prompt:
                # Add user message to chat history (still needed for UI display)
                if prompt:
                    # Create a message dictionary with proper attribute access
                    message = {
                        "role": "user",
                        "raw": prompt,  # Store the whole object if needed
                        "prompt": prompt.text,  # Access text as an attribute
                        "file_name": prompt.files[0].name if prompt.files else None,  # Get file name safely
                        "file_content": None, # Initialize file_content to None
                        "file_tokens": 0, # Initialize file_tokens to 0
                        "prompt_tokens": count_tokens(prompt.text)
                    }
                    st.session_state.prompt_tokens += message["prompt_tokens"]
                    st.session_state.total_tokens_out += message["prompt_tokens"]

                json_data = None  # Initialize json_data to None here, before checking for files

                if prompt.files:
                    uploaded_file = prompt.files[0]
                    # Read the file content
                    file_content = uploaded_file.read()
                    # If it's a JSON file, parse it
                    if uploaded_file.type == 'application/json':
                        json_data = json.loads(file_content)
                        json_tokens = count_tokens(json_data)
                        message["file_tokens"] = json_tokens
                        message["file_content"] = json_data # Store json_data in message
                        st.session_state.file_tokens += message["file_tokens"]
                        st.session_state.total_tokens_out += message["file_tokens"]
                    else:
                        json_data = None # Handle non-JSON files if needed, or just ignore

                st.session_state.messages.append(message)

                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(f"{message['prompt']}")
                    if "file_tokens" in message and message["file_tokens"] > 0 and message["file_name"] is not None:
                        st.markdown(f"File: \"*{message['file_name']}*\" (tokens: {message['file_tokens']:,})")

                # Initialize AI response message placeholder
                with st.chat_message("AI"):
                    response_placeholder = st.empty() # Placeholder for streaming response
                    full_response_content = "" # Accumulate response for token counting
                    st.session_state.chat_generating = True # Disable chat input 

                    try: # Error handling for LLM call
                        # Construct parts for send_message - include text and JSON data
                        parts = [prompt.text] # Start with the text prompt
                        if json_data:
                            # parts.append(json.dumps(json_data, indent=2)) # Add JSON data as a string
                            parts.append(json.dumps(json_data, separators=(',', ":"))) # Add JSON data as a string, without indentations otherwise it becomes too bloated
                           

                        # Use chat_session.send_message() with parts
                        stream_response = chat_session.send_message(
                            parts, # Pass the parts list here
                            stream=True # Request streaming response
                        )
                        last_chunk = None
                        for chunk in stream_response: # Iterate through response chunks
                            if chunk:
                                last_chunk = chunk  # Store the last chunk to access metadata
                                if chunk.text:
                                    full_response_content += chunk.text
                                    response_placeholder.markdown(full_response_content + "▌") # Display chunk with blinking cursor 
                        response_placeholder.markdown(full_response_content) # Final response without cursor

                        # Check for usage metadata in the last chunk
                        has_native_counts = False
                        if last_chunk and hasattr(last_chunk, 'usage_metadata'):
                            usage_metadata = last_chunk.usage_metadata
                            if hasattr(usage_metadata, 'prompt_token_count') and hasattr(usage_metadata, 'candidates_token_count'):
                                has_native_counts = True
                                # Calculate costs once
                                costs = calculate_gemini_cost(
                                    "gemini-2.0-flash", 
                                    usage_metadata.prompt_token_count, 
                                    usage_metadata.candidates_token_count
                                )
                                
                                # Update chat token stats with actual token counts and costs
                                st.session_state.chat_token_stats = {
                                    'total_tokens': st.session_state.chat_token_stats.get('total_tokens', 0) + usage_metadata.total_token_count,
                                    'total_input_tokens': st.session_state.chat_token_stats.get('total_input_tokens', 0) + usage_metadata.prompt_token_count,
                                    'total_output_tokens': st.session_state.chat_token_stats.get('total_output_tokens', 0) + usage_metadata.candidates_token_count,
                                    'prompt_tokens': st.session_state.prompt_tokens,  # Keep existing count for user prompts
                                    'json_tokens': st.session_state.file_tokens,  # Keep existing count for JSON files
                                    'response_tokens': st.session_state.chat_token_stats.get('response_tokens', 0) + usage_metadata.candidates_token_count,
                                    'using_native_counts': True,
                                    'input_cost': st.session_state.chat_token_stats.get('input_cost', 0.0) + costs['input_cost'],
                                    'output_cost': st.session_state.chat_token_stats.get('output_cost', 0.0) + costs['output_cost'],
                                    'total_cost': st.session_state.chat_token_stats.get('total_cost', 0.0) + costs['total_cost']
                                }

                    except Exception as e: # Catch any exceptions during streaming
                        full_response_content = f"Error generating response: {e}"
                        response_placeholder.markdown(full_response_content)
                        has_native_counts = False

                    finally: # Code to execute after try or except
                        st.session_state.chat_generating = False # Re-enable chat input

                # Add assistant response to chat history - store full response content
                estimated_tokens = count_tokens(full_response_content)
                ai_message = {
                    "role": "AI",
                    "response": full_response_content,
                    "response_tokens": estimated_tokens
                }
                
                # If we have native token counts, add them to the message
                if 'has_native_counts' in locals() and has_native_counts:
                    ai_message["native_response_tokens"] = usage_metadata.candidates_token_count
                    ai_message["native_prompt_tokens"] = usage_metadata.prompt_token_count
                    ai_message["native_total_tokens"] = usage_metadata.total_token_count
                    ai_message["using_native_counts"] = True
                else:
                    # Still update the total tokens with estimated count for backward compatibility 
                    st.session_state.total_tokens_in += estimated_tokens
                
                st.session_state.messages.append(ai_message)

        chat_history_col, token_stats_col, download_col, reset_col = st.columns([3,3,3,3])

        with chat_history_col:
            with st.popover("[DEV] Chat history"):
                st.write(st.session_state.messages)
        with token_stats_col:
            with st.popover("Token Stats"):
                # Display token statistics
                if st.session_state.chat_token_stats.get('using_native_counts', False):
                    st.write("**Token Usage (actual):**")
                    stats = st.session_state.chat_token_stats
                    st.markdown(f"- Total Tokens Used: **{stats.get('total_tokens', 0):,}**")
                    st.markdown(f"- User Input: **{stats.get('total_input_tokens', 0):,}** (User Prompts: **{stats.get('prompt_tokens', 0):,}** Files sent: **{stats.get('json_tokens', 0):,}**)")
                    st.markdown(f"- LLM Response: **{stats.get('total_output_tokens', 0):,}**")
                    
                    # Display cost information
                    st.write("**Cost Estimation (USD):**")
                    st.markdown(f"- Input Cost: **${stats.get('input_cost', 0.0):.6f}**")
                    st.markdown(f"- Output Cost: **${stats.get('output_cost', 0.0):.6f}**")
                    st.markdown(f"- Total Cost: **${stats.get('total_cost', 0.0):.6f}**")
                else:
                    st.write("**Token Usage (est):**")
                    st.markdown(f"- Total Tokens Used: **{st.session_state.total_tokens_in + st.session_state.total_tokens_out:,}**")
                    st.markdown(f"- User Input: **{st.session_state.total_tokens_out:,}** (User Prompts: **{st.session_state.prompt_tokens:,}** Files sent: **{st.session_state.file_tokens:,}**)")
                    st.markdown(f"- LLM Response: **{st.session_state.total_tokens_in:,}**")
        with download_col:
            # Generate timestamp for filename in format YYYYMMDD_HHMM
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            chat_filename = f"Chat_{timestamp}.md"
            
            # Create download button for chat history
            if st.session_state.messages:  # Only show if there are messages
                chat_markdown = format_chat_history_as_markdown()
                st.download_button(
                    "Download Chat", 
                    chat_markdown, 
                    file_name=chat_filename,
                    mime="text/markdown"
                )
            else:
                st.button("Download Chat", disabled=True)  # Disabled if no messages 
            
        with reset_col:
            if st.session_state.messages:  # Only show if there are messages
                st.button("Reset Chat", on_click=reset_chat, disabled = False)
            else:
                st.button("Reset Chat", disabled=True)  # Disabled if no messages

    with gen_tab:
        if st.session_state.selected_in_json:
            if "is_report_ready" not in st.session_state:
                st.session_state.is_report_ready = False
                st.session_state.report_generating = False
                st.session_state.report_generated = False
                st.session_state.json_source_option = "Upload JSON File"
                st.session_state.report_token_stats = {}
                st.session_state.reset_report_widgets = False

            if "gen_disabled" not in st.session_state:
                st.session_state.gen_disabled = True

            default_system_prompt = get_default_system_prompt()
            report_sys_prompt = st.text_area("System Prompt", 
                                    value=st.session_state.get("report_log_sys_prompt", ""), 
                                    key="report_sys_prompt", 
                                    disabled=st.session_state.report_generated,
                                    height=min(400, max(70,len(st.session_state.get("report_log_sys_prompt", "").split('\n')) * 25)))

            report_user_prompt = st.text_area("User Prompt", 
                                    value=st.session_state.get("report_log_user_prompt", ""), 
                                    key="report_user_prompt", 
                                    disabled=st.session_state.report_generated,
                                    height=min(400, max(200,len(st.session_state.get("report_log_user_prompt", "").split('\n')) * 25)))

            # JSON Source Selection - Radio Buttons
            st.session_state.json_source_option = st.radio(
                "Choose JSON Data Source:",
                options=["Use Generated JSON","Upload JSON File"],
                index=0,  # Default to "Use Generated JSON"
                key="json_source_radio",
                disabled=st.session_state.report_generated
            )

            if "report_json_uploaded" not in st.session_state:
                report_json_uploaded = None # Initialize to None

            # Handle file uploader with a unique key when reset is triggered
            if st.session_state.json_source_option == "Upload JSON File":
                # Generate a unique key for the file uploader if reset was triggered
                uploader_key = f"report_json_uploaded_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}" if st.session_state.get("reset_report_widgets", False) else "report_json_uploaded"
                
                # Reset the flag after using it
                if st.session_state.get("reset_report_widgets", False):
                    st.session_state.reset_report_widgets = False
                
                report_json_uploaded = st.file_uploader("Upload JSON", type="json", key=uploader_key)
            elif st.session_state.json_source_option == "Use Generated JSON":
                st.info("Using the JSON generated from the selected data.") # Confirmation message
                # No file uploader needed in this case

            gen_disabled = False #(len(st.session_state.report_user_prompt) == 0)
            st.session_state.gen_disabled = gen_disabled

            if "generated_report" not in st.session_state:
                st.session_state.generated_report = ""

            if st.session_state.report_generated: # Display report if it exists
                st.write("---")
                st.subheader("Generated Report:")
                st.markdown(st.session_state.generated_report)

                # Download buttons
                col_dlmd, col_dlpdf, col_dllog, col_tokens, col_reset = st.columns(5)
                with col_dlmd:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    report_filename = f"Report_{timestamp}.md"
                    st.download_button("Download Report (Markdown)", st.session_state.generated_report, file_name=report_filename)
                with col_dlpdf: # PDF download
                    if st.session_state.generated_report: # Only show if report is generated
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                        pdf_filename = f"Report_{timestamp}" # Filename without extension for conversion function
                        pdf_bytes = convert_markdown_to_pdf_python(st.session_state.generated_report, pdf_filename) # Use Python-based conversion
                        if pdf_bytes: # Check if PDF conversion was successful
                            st.download_button(
                                "Download Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"{pdf_filename}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.warning("PDF conversion failed (Python-based). Check error messages above.") # Python-specific message                    else:
                            st.button("Download Report (PDF)", disabled=True) # Disable if no report

                #placeholder for alternative PDF conversion
                # with col_dl2: #  PDF download
                #     if st.session_state.generated_report: # Only show if report is generated
                #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                #         pdf_filename = f"Report_{timestamp}" # Filename without extension for conversion function
                #         pdf_bytes = convert_markdown_to_pdf_opt2(st.session_state.generated_report, pdf_filename) # Use markdown-pdf conversion
                #         if pdf_bytes: # Check if PDF conversion was suc cessful
                #             st.download_button(
                #                 "Download Report (PDF)2",
                #                 data=pdf_bytes,
                #                 file_name=f"{pdf_filename}.pdf",
                #                 mime="application/pdf"
                #             )
                #         else:
                #             st.warning("PDF conversion failed (Python-based). Check error messages above.") # Python-specific message                    else:
                #             st.button("Download Report (PDF)2", disabled=True) # Disable if no report  

                with col_dllog:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    genlog_filename = f"Report_genlog_{timestamp}.md"
                    
                    # Create the report generation log content
                    current_datetime = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S")
                    stats = st.session_state.report_token_stats
                    
                    report_log_content = f"""# Report Generation Log

## Generated: {current_datetime}

## Token Statistics
- Total Tokens Used: **{stats.get('total_tokens', 0):,}**
- User Input: **{stats.get('total_input_tokens', 0):,}** (Prompts: **{stats.get('prompt_tokens', 0):,}**, JSON: **{stats.get('json_tokens', 0):,}**)
- LLM Response: **{stats.get('output_tokens', 0):,}**

## Cost Estimation (USD)
- Input Cost: **${stats.get('input_cost', 0.0):.6f}**
- Output Cost: **${stats.get('output_cost', 0.0):.6f}**
- Total Cost: **${stats.get('total_cost', 0.0):.6f}**

## System Prompt
```
{st.session_state.report_log_sys_prompt}
```

## User Prompt
```
{st.session_state.report_log_user_prompt}
```

---

## Response
{st.session_state.generated_report}
"""
                    
                    st.download_button(
                        "Download Report Generation Log (MD)", 
                        report_log_content, 
                        file_name=genlog_filename
                    )
                
                with col_tokens:
                    with st.popover("Token Stats"):
                        # Display token statistics
                        st.write("**Token Usage (actual):**")
                        stats = st.session_state.report_token_stats
                        st.markdown(f"- Total Tokens Used: **{stats.get('total_tokens', 0):,}**")
                        st.markdown(f"- User Input: **{stats.get('total_input_tokens', 0):,}** ( Prompts: **{stats.get('prompt_tokens', 0):,}**, JSON: **{stats.get('json_tokens', 0):,}**)")
                        st.markdown(f"- LLM Response: **{stats.get('output_tokens', 0):,}**")
                        
                        # Display cost information
                        st.write("**Cost Estimation (USD):**")
                        st.markdown(f"- Input Cost: **${stats.get('input_cost', 0.0):.6f}**")
                        st.markdown(f"- Output Cost: **${stats.get('output_cost', 0.0):.6f}**")
                        st.markdown(f"- Total Cost: **${stats.get('total_cost', 0.0):.6f}**")
                with col_reset:
                    st.button("Reset Report Generation", on_click=reset_report_generation) # Reset button here


            elif st.button("Generate Report", disabled=st.session_state.gen_disabled) and not st.session_state.report_generating: # Add check for report_generating state
                st.session_state.gen_disabled = True
                st.session_state.generated_report = ""
                st.session_state.report_generating = True # Set state to generating
                st.session_state.report_generated = False  # Reset generated state
                st.session_state.report_token_stats = {}     # Reset token stats 

                # Display spinner while generating
                with st.spinner("Generating report..."):
                    # Determine JSON data source based on selection
                    json_data_for_report = None
                    if st.session_state.json_source_option == "Upload JSON File" and report_json_uploaded:
                        json_data_for_report = json.load(report_json_uploaded)
                    elif st.session_state.json_source_option == "Use Generated JSON" and st.session_state.selected_in_json:
                        json_data_for_report = st.session_state.selected_in_json

                    # Connect to Gemini and generate report
                    gemini_model = llm_connect(st.secrets["GEMINI_API_KEY"])

                    # Add instruction to avoid commentary in the system prompt 
                    report_sys_prompt = default_system_prompt+ "\n\n" + report_sys_prompt

                    if gemini_model:
                        # Save the system prompt for the report log
                        # st.session_state.report_log_sys_prompt = report_sys_prompt
                        
                        report_text = generate_llm_response(
                            gemini_model,
                            report_sys_prompt,
                            report_user_prompt,
                            json_data_for_report
                        )
                        st.session_state.generated_report = report_text
                        st.session_state.report_generated = True # Set state to generated
                        st.session_state.rerun_required = True
                        st.toast("Report generated successfully!")

                        # Calculate and store token stats
                        token_data = {
                            "system_prompt": st.session_state.report_log_sys_prompt,
                            "user_prompt": st.session_state.report_log_user_prompt,
                            "json_data": json_data_for_report,
                            "response": report_text
                        }
                        st.session_state.report_token_stats = calculate_all_tokens(
                            prompt = token_data["system_prompt"] + "\n\n---\n\nUser Query: " + token_data["user_prompt"],
                            json_data = token_data["json_data"],
                            response = token_data["response"]
                        )
                
                st.session_state.report_generating = False # Reset generating state
                # st.session_state.gen_disabled = False # Re-enable button

        if st.session_state.get("report_generated", False) and st.session_state.get("rerun_required", False):
            print("inside rerun_required check")
            st.session_state.rerun_required = False
            st.rerun()   

if __name__ == "__main__":
    main()
