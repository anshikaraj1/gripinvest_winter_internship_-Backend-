import pandas as pd

# Load raw NASA dataset
df = pd.read_csv('nasa_raw_data.txt', sep=' ', header=None)

# Drop blank columns caused by double spaces
df.dropna(axis=1, how='all', inplace=True)

# Assign proper column names
column_names = ['unit_number', 'time_in_cycles', 'operational_setting_1', 'operational_setting_2',
                'operational_setting_3', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
                'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11',
                'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17',
                'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

df.columns = column_names

# Calculate RUL for each engine
rul = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul.columns = ['unit_number', 'max_cycle']
df = df.merge(rul, on='unit_number')
df['RUL'] = df['max_cycle'] - df['time_in_cycles']
df.drop('max_cycle', axis=1, inplace=True)

# Save cleaned dataset
df.to_csv('nasa_cleaned.csv', index=False)
print("NASA dataset cleaned and saved as nasa_cleaned.csv")
