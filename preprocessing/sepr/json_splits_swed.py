import pandas as pd
import argparse
import json

def create_json(df, in_col, out_col, instruction, out_name):
    # Create json with alpaca prompt format
    docs = []
    for index, row in df.iterrows():
        doc = {
            "instruction": instruction,
            "input": row[in_col],
            "output": row[out_col],
            "id": row["_id"]
        }
        docs.append(doc)
    with open(out_name, 'w') as f:
        json.dump(docs, f, indent=0)

def main(notes_file, splits_file):
    notes = pd.read_feather(notes_file)
    splits = pd.read_feather(splits_file)

    train_ids = splits.loc[splits['split'] == 'train', '_id'].tolist()
    test_ids = splits.loc[splits['split'] == 'test', '_id'].tolist()
    val_ids = splits.loc[splits['split'] == 'val', '_id'].tolist()

    # Filter the df for each split
    filtered_train = notes[notes['_id'].isin(train_ids)]
    filtered_test = notes[notes['_id'].isin(test_ids)]
    filtered_val = notes[notes['_id'].isin(val_ids)]

    inst = "Utifrån en lista med textuella beskrivningar av diagnoskoder, generera en motsvarande klinisk anteckning som ger omfattande och relevanta detaljer om patientens sjukdomshistoria, nuvarande tillstånd och behandling som mottagits på sjukhuset."

    # Create JSON files for each split
    create_json(filtered_train, in_col='target_transcriptions', out_col='text', instruction=inst, out_name="sepr_l.json")
    create_json(filtered_test, in_col='target_transcriptions', out_col='text', instruction=inst, out_name="sepr_m.json")
    create_json(filtered_val, in_col='target_transcriptions', out_col='text', instruction=inst, out_name="sepr_s.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes_file", type=str, required=True, help="Path to the notes Feather file.")
    parser.add_argument("--splits_file", type=str, required=True, help="Path to the splits Feather file.")
    args = parser.parse_args()

    main(args.notes_file, args.splits_file)
