import pandas as pd
import argparse
from collections import Counter

def analyze_predictions(file, threshold, target_column='target', id_column='_id'):
    # Analyze predictions of medical coding model
    df = pd.read_feather(file)
    
    df = df.drop(columns=[id_column])
    
    # Create a binary prediction df based on the threshold
    predicted_codes = df.drop(columns=[target_column]).applymap(lambda x: x >= threshold)
    
    predicted_code_lists = predicted_codes.apply(lambda row: row.index[row].tolist(), axis=1)
    
    avg_predicted_length = predicted_code_lists.apply(len).mean()
    avg_target_length = df[target_column].apply(len).mean()
    
    total_predicted_codes = sum(predicted_code_lists.apply(len))
    total_actual_codes = sum(df[target_column].apply(len))
    
    predicted_code_counter = Counter([code for codes in predicted_code_lists for code in codes])
    target_code_counter = Counter([code for codes in df[target_column] for code in codes])
    
    all_target_codes = set(code for codes in df[target_column] for code in codes)
    correctly_predicted_codes = all_target_codes.intersection(predicted_code_counter.keys())
    num_correctly_predicted_codes = len(correctly_predicted_codes)
    
    total_unique_target_codes = len(all_target_codes)

    unique_predicted_codes = len(predicted_code_counter)

    percentage_correctly_predicted = (num_correctly_predicted_codes / total_unique_target_codes) * 100 if total_unique_target_codes > 0 else 0

    # Store results in a dict for display
    results = {
        "Average Length of Predicted Codes": avg_predicted_length,
        "Average Length of Target Codes": avg_target_length,
        "Total Predicted Codes": total_predicted_codes,
        "Total Actual Codes": total_actual_codes,
        "Number of Correctly Predicted Codes": num_correctly_predicted_codes,
        "Total Unique Target Codes": total_unique_target_codes,
        "Unique Predicted Codes": unique_predicted_codes, 
        "Percentage of Correctly Predicted Codes": percentage_correctly_predicted,
    }
    
    # print
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the predictions file.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for classification.")
    parser.add_argument("--target_column", type=str, default="target", help="Name of the target column in the file.")
    parser.add_argument("--id_column", type=str, default="_id", help="Name of the ID column to exclude.")
    
    args = parser.parse_args()

    analyze_predictions(
        file=args.file,
        threshold=args.threshold
    )