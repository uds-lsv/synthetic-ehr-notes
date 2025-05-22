import pandas as pd
from sklearn.metrics import f1_score
import argparse

def get_doc_length(train_path, split_path):
    # create dictionary with doc lenghts and ids 
    train = pd.read_feather(train_path)
    splits = pd.read_feather(split_path)
    train_ids = splits[splits["split"] == "test"]['_id']
    filtered = train[train['_id'].isin(train_ids)]
    doc_lengths = {
        row['_id']: len(str(row['text']).split()) 
        for _, row in filtered.iterrows()
    }
    return doc_lengths

def calculate_f1_per_document(df, length_dict, threshold=0.5):
    # create df storing f1, id and length for each docuement

    codes = df.columns.difference(['_id', 'target'])  # Get the code columns (features)
    
    results = []

    # Get predictions
    binary_predictions = df[codes] > threshold

    for idx, row in df.iterrows():
        actual = [1 if code in row['target'] else 0 for code in codes]
        
        predicted = binary_predictions.loc[idx].astype(int).values
        
        f1 = f1_score(actual, predicted, zero_division=0)
        
        doc_length = length_dict.get(row['_id'], 0)  
        
        results.append({
            '_id': row['_id'],  
            'f1': f1,
            'doc_length': doc_length  
        })

    f1_df = pd.DataFrame(results)
    
    return f1_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate F1 scores per document and save results.")
    parser.add_argument("--train_path", required=True, help="Path to the training data Feather file.")
    parser.add_argument("--split_path", required=True, help="Path to the splits Feather file.")
    parser.add_argument("--pred_file", required=True, help="Path to the predictions Feather file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification.")
    parser.add_argument("--output_file", required=True, help="Path to save the output CSV file.")
    
    args = parser.parse_args()

    # Get doc lengths
    doc_lengths = get_doc_length(args.train_path, args.split_path, args.field)

    # Load predictions
    predictions = pd.read_feather(args.predictions_path)

    # Calculate F1 per document
    f1_df = calculate_f1_per_document(predictions, doc_lengths, threshold=args.threshold)

    # Save the results to csv
    f1_df.to_csv(args.output_file, index=False)