# Bank Statement Network Analyzer

A desktop application that analyzes bank statements from Excel files and creates force-directed network visualizations of transactions between senders and recipients.

## Features
- Import bank statements from Excel files
- Visualize transaction networks using force-directed graphs
- Interactive network visualization
- Automatic clustering of common senders and recipients

## Requirements
- Python 3.8 or higher
- Required packages listed in requirements.txt

## Installation
1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python main.py
   ```
2. Click "Upload Excel File" to select your bank statement
3. The application will automatically generate a network visualization

## Excel File Format
Your Excel file should contain at least these columns:
- Sender: Name or ID of the transaction sender
- Recipient: Name or ID of the transaction recipient

## Note
This is a basic version of the application. Future updates may include:
- Transaction amount visualization
- Time-based filtering
- Advanced network analytics
- Custom layout options
