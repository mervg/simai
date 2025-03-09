# SIMAI - SalesIM AI Analysis Engine

## Overview
SIMAI (SalesIM AI Analysis Engine) is a modular AI-driven platform for data analysis. This repository contains the web interface component that allows internal stakeholders to interact with UDF (Unified Data Format) data, implement filtering, and export functionality.

## Current Version
This repository currently contains a test application (`simai_test_app.py`) to verify the deployment workflow. The full MVP 1.0 functionality will be implemented in subsequent updates.

## Features
- **Test Application**: Interactive Streamlit demo with data visualization, widgets, and progress indicators
- **Coming in MVP 1.0**:
  - Select UDF databases
  - Filter UDF records by parameters
  - Export filtered data to JSON
- **Coming in Future Versions**:
  - Run prompts through the Gemini 2.0 API
  - Save and display reports

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# To run the test application
streamlit run simai_test_app.py

# To run the main application (coming soon)
streamlit run simai_app.py
```

## Project Status
This project is in active development. The current focus is on testing the deployment workflow with a simple Streamlit application before implementing the full MVP 1.0 functionality.

## License
Internal use only.
