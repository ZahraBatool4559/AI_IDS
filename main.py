import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("data/dataset.csv", encoding="latin1")

# Step 2: Print the first 5 rows
print(df.head())

# Step 3: Print information about the dataset
print(df.info())

# Step 4: Print basic statistics
print(df.describe())
