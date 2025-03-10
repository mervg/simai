import streamlit as st
import pandas as pd
import os
import sqlite3
import json
from typing import Dict, List, Tuple, Any 

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

SQLITE_DB_PATH = os.path.dirname(__file__)
SQLITE_DB_FILENAME = os.path.join(SQLITE_DB_PATH, 'UDF.db')

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

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    # temporarily disabled as it was pollution concole - need to deal with it later
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

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
                    df = df[df[column].astype(str).str.contains(user_text_input)]

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


st_init_page() #called outside of main() to not be run once and not be included into reruns

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
        st.write(f"\nSelected records: **{st.session_state.selected_df_recs}**")

    st.dataframe(st.session_state.selected_df, hide_index=True, use_container_width=True)

    # JSON generation
    if "selected_in_json" not in st.session_state:
        st.session_state.selected_in_json = {}

    col21, col22 = st.columns([1,5])

    if st.session_state.selected_df_recs > 0:
        with col21:
            st.button("Generate JSON", on_click=generate_json_of_selected)

    if st.session_state.selected_in_json:
        with col22:
            st.write("\nGenerated JSON:")
            st.json(st.session_state.selected_in_json, expanded=3)

    if "json_filename" not in st.session_state:
        st.session_state.json_filename = "UDF_selected.json"

    if st.session_state.selected_in_json:
        with col21:
            st.session_state.json_filename = st.text_input("JSON filename", value=st.session_state.json_filename)
        with col21:
            st.download_button("Download JSON", json.dumps(st.session_state.selected_in_json, indent=2), st.session_state.json_filename)


if __name__ == "__main__":
    main()


