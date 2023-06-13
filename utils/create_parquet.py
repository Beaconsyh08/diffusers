import pandas as pd

# Create a DataFrame with your data
data = {
    'Column1': [1, 2, 3, 4, 5],
    'Column2': ['A', 'B', 'C', 'D', 'E'],
    'Column3': [True, False, True, True, False]
}

df = pd.DataFrame(data)

# Save the DataFrame as a Parquet file
df.to_parquet('./file.parquet')
