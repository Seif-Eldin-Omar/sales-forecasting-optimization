import pandas as pd
from preprocessing import preprocess

def main():
    """
    Main function to load raw data, preprocess it, and save the cleaned data.
    
    This function serves as the entry point for the script.
    It loads the raw data from a specified path, applies preprocessing steps,
    and saves the cleaned data to a new CSV file.
    """

    raw_data_path = 'data/explored_train.csv'

    df = pd.read_csv(raw_data_path, parse_dates=['date'])

    clean_df = preprocess(df)

    output_path = 'data/cleaned_data.csv'
    clean_df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")
    print(f"Final shape: {clean_df.shape}")

if __name__ == '__main__':
    main()