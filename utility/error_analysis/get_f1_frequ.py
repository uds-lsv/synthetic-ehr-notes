import pandas as pd
from sklearn.metrics import f1_score
from collections import Counter
import argparse

def get_frequ(train_path, split_path, field):
    # compute code frequencies in training data
    train = pd.read_feather(train_path)
    splits = pd.read_feather(split_path)
    train_ids = splits[splits["split"] == "train"]['_id']
    filtered = train[train['_id'].isin(train_ids)]
    all_codes = [code for sublist in filtered['target'] for code in sublist]
    code_frequencies = dict(Counter(all_codes))
    return code_frequencies

def calculate_f1_per_code(pred_file, freq_dict, threshold):
    # calculate f1 for each code
    df = pd.read_feather(pred_file)
    codes = df.columns.difference(['_id', 'target'])  
    
    results = []

    binary_predictions = df[codes] > threshold

    for code in codes:
        actual = df['target'].apply(lambda x: code in x).astype(int)
        predicted = binary_predictions[code].astype(int)
        
        f1 = f1_score(actual, predicted, zero_division=0)
        
        results.append({
            'code': code,
            'f1': f1,
            'frequency': freq_dict.get(code, 0)  
        })

    f1_df = pd.DataFrame(results)
    
    return f1_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data Feather file.")
    parser.add_argument("--split_path", type=str, required=True, help="Path to the split Feather file.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the predictions Feather file.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for classification.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    
    args = parser.parse_args()

    # Get frequency of codes in the training data
    code_freqs = get_frequ(args.train_path, args.split_path, args.field)

    # Calculate F1 scores for each code
    f1_df = calculate_f1_per_code(args.pred_file, code_freqs, args.threshold)

    # Save to csv file
    f1_df.to_csv(args.output_file, index=False)