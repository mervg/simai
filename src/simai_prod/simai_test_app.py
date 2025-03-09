"""
Simple Streamlit Test Application for SIMAI Project.

This application serves as a proof of concept for the SIMAI deployment workflow.
It demonstrates basic Streamlit features and interactive elements.

Usage:
    streamlit run simai_test_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Basic page configuration
st.set_page_config(
    page_title="SIMAI Test App TEST",
    page_icon="🧪"
)

# Main header
st.title("SIMAI Test Application TEST")

# Introduction section
st.header("Welcome to SIMAI!")
st.write(
    "This is a simple test application to verify the deployment workflow "
    "for the SIMAI project. It demonstrates various Streamlit features that will be used in the "
    "full application."
)

# Current time display
st.subheader("Current Time")
st.write(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar configuration
st.sidebar.title("Test Controls")
st.sidebar.write("Explore different interactive elements")

# Demo section selection in sidebar
demo_section = st.sidebar.radio(
    "Select Demo Section:",
    ["Data Visualization", "Interactive Widgets", "Progress Indicators"]
)

# Data Visualization Demo
if demo_section == "Data Visualization":
    st.header("Data Visualization Demo")
    
    # Generate sample data
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Line Chart", "Bar Chart", "Area Chart"]
    )
    
    # Sample data generation
    dates = pd.date_range(start='2025-01-01', periods=20, freq='D')
    data = np.random.randn(20, 3).cumsum(axis=0)
    df = pd.DataFrame(data, columns=['Product A', 'Product B', 'Product C'], index=dates)
    
    # Display different chart types based on selection
    if chart_type == "Line Chart":
        st.line_chart(df)
    elif chart_type == "Bar Chart":
        st.bar_chart(df)
    else:
        st.area_chart(df)
    
    # Data table display
    st.subheader("Sample Data Table")
    st.dataframe(df)

# Interactive Widgets Demo
elif demo_section == "Interactive Widgets":
    st.header("Interactive Widgets Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Widgets")
        user_name = st.text_input("Enter your name:", "Guest")
        user_age = st.slider("Select your age:", 18, 100, 30)
        user_date = st.date_input("Select a date:")
        
        if st.button("Submit Information"):
            st.success(f"Hello {user_name}! You are {user_age} years old and selected {user_date}.")
    
    with col2:
        st.subheader("Selection Widgets")
        options = st.multiselect(
            "Select your favorite colors:",
            ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"],
            ["Blue"]
        )
        
        st.write("You selected:", ", ".join(options))
        
        agree = st.checkbox("I agree to the terms and conditions")
        if agree:
            st.info("Thank you for agreeing to the terms!")

# Progress Indicators Demo
else:
    st.header("Progress Indicators Demo")
    
    # Progress bar demo
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if st.button("Start Process"):
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Process running: {i}% complete")
            time.sleep(0.05)
        status_text.success("Process completed successfully!")
    
    # Spinner demo
    with st.expander("Spinner Demo"):
        if st.button("Run Background Task"):
            with st.spinner("Running background task..."):
                time.sleep(3)
            st.success("Background task completed!")

# Footer
st.markdown("---")
st.markdown(
    "SIMAI Test Application | Created for deployment workflow testing | 2025"
)
