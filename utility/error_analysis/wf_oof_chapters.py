import argparse
import pandas as pd
import matplotlib.pyplot as plt

blocks = {
    'I': ['a00', 'b99'],
    'II': ['c00', 'd48'],
    'III': ['d50', 'd89'],
    'IV': ['e00', 'e90'],
    'V': ['f00', 'f99'],
    'VI': ['g00', 'g99'],
    'VII': ['h00', 'h59'],
    'VIII': ['h60', 'h95'],
    'IX': ['i00', 'i99'],
    'X': ['j00', 'j99'],
    'XI': ['k00', 'k93'],
    'XII': ['l00', 'l99'],
    'XIII': ['m00', 'm99'],
    'XIV': ['n00', 'n99'],
    'XV': ['o00', 'o99'],
    'XVI': ['p00', 'p96'],
    'XVII': ['q00', 'q99'],
    'XVIII': ['r00', 'r99'],
    'XIX': ['s00', 't98'],
    'XX': ['v01', 'y98'],
    'XXI': ['z00', 'z99'],
    'XXII': ['u00', 'u99']
}
color_mapping = {
    "I": "#1f77b4",     # Blue
    "II": "#ff6e00",    # Orange
    "III": "#2ca02c",   # Green
    "IV": "#ff3333",    # Red
    "V": "#a50b5e",     # Purple
    "VI": "#8c564b",    # Brown
    "VII": "#e377c2",   # Pink
    "VIII": "#7f7f7f",  # Gray
    "IX": "#33ff88",    # Olive
    "X": "#17becf",     # Teal
    "XI": "#ffc133",    # Light Blue
    "XII": "#ffbb78",   # Light Orange
    "XIII": "#98df8a",  # Light Green
    "XIV": "#ff9896",   # Salmon
    "XV": "#c5b0d5",    # Lavender
    "XVI": "#c49c94",   # Beige
    "XVII": "#f7b6d2",  # Light Pink
    "XVIII": "#ff33c3", # Light Gray
    "XIX": "#a8ff33",   # Light Olive
    "XX": "#9edae5",    # Light Teal
    "XXI": "#b833ff",   # Deep Red
    "XXII": "#8c6d31",  # Dark Brown
    "Others": "#c7c7c7"       # Light Gray for "Others"
}


def family_errors(preds, blocks, plotname, threshold=0.5):
    # Create a mapping from code prefix to block
    code_to_block = {}
    for block, (start, end) in blocks.items():
        start_letter = start[0]
        end_letter = end[0]
        start_num = int(start[1:])
        end_num = int(end[1:])
        for i in range(start_num, end_num + 1):
            code = f"{start_letter}{i:0>2}"
            code_to_block[code] = block
        for letter in range(ord(start_letter) + 1, ord(end_letter) + 1):
            for i in range(100):
                code = f"{chr(letter)}{i:0>2}"
                if chr(letter) + str(i).zfill(2) <= end:
                    code_to_block[code] = block

    logits_columns = preds.columns.difference(['_id', 'target']) 
    binary_predictions = preds[logits_columns] > threshold
    predictions_df = pd.DataFrame(binary_predictions, columns=logits_columns)

    all_predictions_counts = {block: 0 for block in blocks.keys()}
    total_predictions = 0
    total_within_family_errors = 0
    total_out_of_family_errors = 0
    total_correct_predictions = 0

    for index, row in predictions_df.iterrows():
        actual_codes = set(code.lower() for code in preds['target'][index])  
        predicted_codes = [col.lower() for col, pred in row.items() if pred == 1]
        total_predictions += len(predicted_codes)
        
        for code in predicted_codes:
            if code[:3] in code_to_block:
                all_predictions_counts[code_to_block[code[:3]]] += 1
            if code in actual_codes:
                total_correct_predictions += 1
            else:
                code_block = code_to_block.get(code[:3])
                in_family = any(code_to_block.get(actual_code[:3]) == code_block for actual_code in actual_codes)
                if in_family:
                    total_within_family_errors += 1
                else:
                    total_out_of_family_errors += 1

    print(f"Total Within Family Errors: {total_within_family_errors}")
    print(f"Total Out of Family Errors: {total_out_of_family_errors}")
    print(f"Total Correct Predictions: {total_correct_predictions}")
    print(f"Total Predicted Codes (Correct and Incorrect): {total_predictions}")

    # Create DataFrame for counts and calculate percentage
    block_counts_df = pd.DataFrame(list(all_predictions_counts.items()), columns=['Block', 'Count'])
    total_count = block_counts_df['Count'].sum()
    block_counts_df['Percentage'] = block_counts_df['Count'] / total_count * 100

    # Add 'Others' for categories < 2%
    others_count = block_counts_df[block_counts_df['Percentage'] < 2]['Count'].sum()
    if others_count > 0:
        others_row = pd.DataFrame([['Others', others_count]], columns=['Block', 'Count'])
        block_counts_df = block_counts_df[block_counts_df['Percentage'] >= 2]
        block_counts_df = pd.concat([block_counts_df, others_row], ignore_index=True)
    else:
        block_counts_df = block_counts_df[block_counts_df['Percentage'] >= 2]

    colors = [color_mapping[block] for block in block_counts_df['Block']]

    # Plot pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(
        block_counts_df['Count'], 
        labels=block_counts_df['Block'], 
        autopct='%1.1f%%', 
        startangle=140,
        textprops={'fontsize': 6},
        colors = colors
    )
    plt.title('Prediction Code Distribution: Synthetic', fontsize=14)
    plt.axis('equal')
    plt.savefig(plotname)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True, help="Path to the predictions file (Feather format).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification.")
    parser.add_argument("--plot_output", type=str, required=True, help="Path to save the output plot.")
    args = parser.parse_args()

    # Load predictions
    preds = pd.read_feather(args.preds)

    # Run family errors analysis
    family_errors(preds, blocks, args.plot_output, args.threshold)
