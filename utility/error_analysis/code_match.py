import pandas as pd
import argparse

def copied_sequences(pred_file, data_file, split_file, threshold):
    # Calculate percentage of predictions that are identically present in the training targets
    # or are a subset of a code combination in the training targets
    pred_df = pd.read_feather(pred_file)
    data_df = pd.read_feather(data_file)
    split_df = pd.read_feather(split_file)

    # Get training df to extract target
    train_ids = split_df.loc[split_df["split"] == "train", '_id']
    train_df = data_df[data_df['_id'].isin(train_ids)]
    train_codes = {frozenset(row) for row in train_df["target"]}  
    
    # Extract predictions
    logits_columns = pred_df.columns.difference(['_id', 'target'])
    binary_predictions = pred_df[logits_columns] > threshold
    
    copied = 0
    subset_copied = 0
    
    for idx, row in binary_predictions.iterrows():
        predicted_codes = frozenset([col for col, pred in row.items() if pred == 1])
        
        if predicted_codes in train_codes:
            copied += 1
            
        elif any(predicted_codes.issubset(train_code) for train_code in train_codes):
            subset_copied += 1

    # Calculate the (subset) matches
    copied_percentage = (copied / len(pred_df)) * 100 if len(pred_df) > 0 else 0
    subset_copied_percentage = (subset_copied / len(pred_df)) * 100 if len(pred_df) > 0 else 0
    
    # Print 
    print(f"Exact Match Percentage: {copied_percentage:.2f}%")
    print(f"Subset Match Percentage: {subset_copied_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the predictions Feather file.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data Feather file containing the training targets.")
    parser.add_argument("--split_file", type=str, required=True, help="Path to the split Feather file containing splits.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for classification.")
    
    args = parser.parse_args()

    copied_sequences(
        pred_file=args.pred_file,
        data_file=args.data_file,
        split_file=args.split_file,
        threshold=args.threshold
    )