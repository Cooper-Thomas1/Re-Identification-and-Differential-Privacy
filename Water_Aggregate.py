import pandas as pd

# Read the CSV file
df = pd.read_csv('water_data.csv', delimiter=';')

# Convert 'datetime' column to datetime format
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

# Set 'datetime' column as the index
df.set_index('datetime', inplace=True)

# Resample data into daily intervals and calculate the average meter reading for each day

df_aggregated = df.resample('D').mean()

# Reset index to make 'datetime' and 'user.key' columns again
df_aggregated = df_aggregated.reset_index()

# Write aggregated data to a new CSV file
df_aggregated.to_csv('output_aggregated_daily.csv', index=False)
print("Aggregated data saved to 'output_aggregated_daily.csv'")
