import pandas as pd
import argparse

def process_icd_data(icd_file, mimic_file, output_file):
    icd_codes = pd.read_csv(icd_file)

    # Filter for ICD-10 codes
    icd10_codes = icd_codes[icd_codes["icd_version"] == 10]

    mimic_file = pd.read_feather(mimic_file)

    icd_column = mimic_file["target"]
    all_transcriptions = []
    for icd_list in icd_column:
        icd_list = icd_list.strip("[]").split()
        icd_list = [item.replace("'", "").replace(".", "") for item in icd_list]
        
        # Get transcriptions for each code
        transcriptions = []
        for code in icd_list:
            transcription = icd10_codes.loc[icd10_codes['icd_code'] == code, "long_title"].values
            if len(transcription) > 0:
                transcriptions.append(transcription[0])
        all_transcriptions.append(transcriptions)

    # Add the transcriptions to mimic file
    mimic_file["target_transcriptions"] = all_transcriptions

    mimic_file.to_feather(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--icd_file", type=str, required=True, help="Path to file with ICD-10 descriptions.")
    parser.add_argument("--mimic_file", type=str, required=True, help="Path to the input file (Feather format.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file (Feather format).")
    args = parser.parse_args()

    process_icd_data(args.icd_file, args.mimic_file, args.output_file)
