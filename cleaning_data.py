import pandas as pd
import numpy as np

def clean_and_rename_vehicle_data(file_path):
    """
    Clean vehicle telemetry data and rename columns to match the desired header format
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # 1. Remove completely empty rows (where all values are NaN)
    df_cleaned = df.dropna(how='all')
    print(f"After removing completely empty rows: {df_cleaned.shape}")
    
    # 2. Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"After removing duplicates: {df_cleaned.shape}")
    
    # 3. Handle time column - convert to datetime and remove date part
    df_cleaned['time'] = pd.to_datetime(df_cleaned['time'], errors='coerce')
    
    # Extract only the time part (remove date)
    df_cleaned['time'] = df_cleaned['time'].dt.time
    
    # 4. Create mapping from original columns to desired column names
    column_mapping = {
        'time': 'Time',
        'Vehicle speed (km/h)': 'Vehicle Speed Sensor [km/h]',
        'Engine RPM (rpm)': 'Engine RPM [RPM]',
        'Absolute pedal position D (%)': 'Accelerator Pedal Position D [%]',
        'Absolute throttle position B (%)': 'Absolute Throttle Position [%]',
        'Engine Fuel Rate (g/sec)': 'Air Flow Rate from Mass Flow Sensor [g/s]'
    }
    
    # 5. Rename columns that exist in the original data
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_cleaned.columns}
    df_cleaned = df_cleaned.rename(columns=existing_columns)
    
    print(f"Renamed columns: {existing_columns}")
    
    # 6. Create the desired header structure with ONLY the columns you need
    desired_columns = [
        'Time',
        'Vehicle Speed Sensor [km/h]',
        'Engine RPM [RPM]',
        'Accelerator Pedal Position D [%]',
        'Absolute Throttle Position [%]',
        'Air Flow Rate from Mass Flow Sensor [g/s]'
    ]
    
    # 7. Create a new dataframe with the desired column structure
    final_df = pd.DataFrame(columns=desired_columns)
    
    # 8. Copy data from cleaned dataframe to final dataframe for columns that exist
    for col in desired_columns:
        if col in df_cleaned.columns:
            final_df[col] = df_cleaned[col]
        else:
            # Add empty column if it doesn't exist in original data
            final_df[col] = np.nan
            print(f"Note: Column '{col}' not found in original data - created as empty")
    
    # 9. Convert numeric columns, handling empty strings and non-numeric values
    numeric_columns = [col for col in desired_columns if col != 'Time']
    for col in numeric_columns:
        if col in final_df.columns:
            # Replace empty strings with NaN
            final_df[col] = final_df[col].replace('', np.nan)
            # Convert to numeric, coercing errors to NaN
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    
    # 10. Sort by time to ensure chronological order
    if 'Time' in final_df.columns:
        final_df = final_df.sort_values('Time').reset_index(drop=True)
    
    # 11. Remove rows where all sensor data is missing (keeping only time is not useful)
    sensor_columns = [col for col in final_df.columns if col != 'Time']
    final_df = final_df.dropna(subset=sensor_columns, how='all')
    
    print(f"\nFinal data shape: {final_df.shape}")
    print(f"Final columns: {final_df.columns.tolist()}")
    
    return final_df

def analyze_missing_data(df):
    """Analyze and report on missing data"""
    print("\n=== MISSING DATA ANALYSIS ===")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percentage
    })
    
    print(missing_info[missing_info['Missing Count'] > 0])
    
    return missing_info

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")

# Usage
if __name__ == "__main__":
    input_file = "data_filled.csv"
    output_file = "cleaned_vehicle_data.csv"  # Simple filename without date
    
    # Clean the data and rename columns
    cleaned_df = clean_and_rename_vehicle_data(input_file)
    
    # Display basic info about cleaned data
    print("\n=== CLEANED DATA INFO ===")
    print(cleaned_df.info())
    
    # Analyze missing data
    analyze_missing_data(cleaned_df)
    
    print("\n=== FIRST 10 ROWS ===")
    print(cleaned_df.head(10))
    
    # Save cleaned data
    save_cleaned_data(cleaned_df, output_file)
    
    # Display summary statistics for available data
    print("\n=== DATA SUMMARY ===")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(cleaned_df[numeric_cols].describe())
    
    # Show data availability
    print("\n=== DATA AVAILABILITY ===")
    for col in cleaned_df.columns:
        available_data = cleaned_df[col].notna().sum()
        total_rows = len(cleaned_df)
        percentage = (available_data / total_rows) * 100
        print(f"{col}: {available_data}/{total_rows} ({percentage:.1f}%)")