import argparse
import pandas as pd
import random
from nltk.corpus import words
import nltk

nltk.download('words')

random.seed(42)

def add_noise_to_feather_h1(file_path, split_path, noise_percentage, output_file_path):
    # Adds noise to each document buy subsituting a specified percentage with random words

    data = pd.read_feather(file_path)
    splits = pd.read_feather(split_path)
    ids = splits.loc[splits["split"].isin(["train", "val"]), '_id']
    
    # Filter df for training and validation rows
    df = data[data['_id'].isin(ids)].copy()
    
    # Generate a pool of random words from the nltk dictionary
    random_word_pool = words.words()

    noisy_documents = []
    for doc in df['text']:
        words_in_doc = doc.split() 
        num_words_to_substitute = int(len(words_in_doc) * (noise_percentage / 100))
        
        indices_to_substitute = random.sample(range(len(words_in_doc)), num_words_to_substitute)
        
        for idx in indices_to_substitute:
            words_in_doc[idx] = random.choice(random_word_pool)
        
        noisy_documents.append(' '.join(words_in_doc))
    
    df['text'] = noisy_documents
    
    # Merge noisy df to original df
    combined_df = data.copy()  
    combined_df.update(df)     

    combined_df.to_feather(output_file_path)
    print(f"Noisy file saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input Feather file.")
    parser.add_argument("--split_path", type=str, required=True, help="Path to the split file with IDs.")
    parser.add_argument("--noise_percentage", type=float, required=True, help="Percentage of words to substitute (0-100).")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the modified Feather file.")
    args = parser.parse_args()

    add_noise_to_feather_h1(args.file_path, args.split_path, args.noise_percentage, args.output_file_path)