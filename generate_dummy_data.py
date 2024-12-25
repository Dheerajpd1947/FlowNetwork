import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# List of banks and companies for realistic data
banks = [
    "Chase Bank", "Bank of America", "Wells Fargo", "Citibank", "Capital One",
    "TD Bank", "US Bank", "PNC Bank", "HSBC", "Santander",
    "Goldman Sachs", "Morgan Stanley", "Deutsche Bank", "Barclays", "UBS"
]

companies = [
    "Amazon", "Walmart", "Target", "Costco", "Best Buy",
    "Apple", "Microsoft", "Google", "Facebook", "Netflix",
    "Uber", "DoorDash", "Grubhub", "Instacart", "Airbnb"
]

# Combine banks and companies for sender/recipient pool
entities = banks + companies

# Generate random dates within the last year
end_date = datetime(2024, 12, 12)
start_date = end_date - timedelta(days=365)
random_days = np.random.randint(0, 365, 1000).tolist()
dates = [start_date + timedelta(days=int(x)) for x in random_days]

# Function to generate random account numbers
def generate_account_number():
    return f"{random.randint(100000000, 999999999):09d}"

# Dictionary to store account numbers for entities
entity_accounts = {entity: generate_account_number() for entity in entities}

# Initialize lists for transaction data
dates_list = dates
senders_list = []
recipients_list = []
amounts_list = []
transaction_ids = []
primary_names = []
primary_numbers = []

# Generate transactions
for i in range(1000):
    # Select random sender and recipient
    sender = random.choice(entities)
    recipient = random.choice([e for e in entities if e != sender])
    
    # Decide if primary account should match sender or recipient
    if random.choice([True, False]):
        primary_name = sender
        primary_number = entity_accounts[sender]
    else:
        primary_name = recipient
        primary_number = entity_accounts[recipient]
    
    senders_list.append(sender)
    recipients_list.append(recipient)
    amounts_list.append(round(np.random.lognormal(mean=8, sigma=1), 2))
    transaction_ids.append(f'TXN{i:06d}')
    primary_names.append(primary_name)
    primary_numbers.append(primary_number)

# Create the data dictionary
data = {
    'Date': dates_list,
    'PrimaryAccountName': primary_names,
    'PrimaryAccountNumber': primary_numbers,
    'Sender': senders_list,
    'SenderAccountNumber': [entity_accounts[sender] for sender in senders_list],
    'Recipient': recipients_list,
    'RecipientAccountNumber': [entity_accounts[recipient] for recipient in recipients_list],
    'Amount': amounts_list,
    'TransactionID': transaction_ids
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by date
df = df.sort_values('Date')

# Save to Excel
output_file = 'dummy_bank_statement.xlsx'
df.to_excel(output_file, index=False)
print(f"Created dummy bank statement with {len(df)} transactions in {output_file}")

# Print some statistics
print("\nData Statistics:")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Average Transaction Amount: ${df['Amount'].mean():.2f}")
print(f"Total Transaction Amount: ${df['Amount'].sum():.2f}")

# Verify primary account matches
sender_matches = len(df[df['PrimaryAccountName'] == df['Sender']])
recipient_matches = len(df[df['PrimaryAccountName'] == df['Recipient']])
print("\nPrimary Account Matching:")
print(f"Matches with Sender: {sender_matches} transactions ({sender_matches/len(df)*100:.1f}%)")
print(f"Matches with Recipient: {recipient_matches} transactions ({recipient_matches/len(df)*100:.1f}%)")
print(f"Total Transactions: {len(df)}")

# Print some sample transactions
print("\nSample Transactions:")
sample_df = df[['Date', 'PrimaryAccountName', 'PrimaryAccountNumber', 
                'Sender', 'SenderAccountNumber', 
                'Recipient', 'RecipientAccountNumber', 'Amount']].head()
print(sample_df.to_string())
