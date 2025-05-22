import argparse
import pandas as pd
import random
from nltk.corpus import words
import nltk

nltk.download('words')

random.seed(40)

def add_noise_to_feather(file_path, split_path, noise_percentage, output_file_path):
    # Add 90% of noise to a sepcified percentage of documents withing the df by subsitution with random words

    data = pd.read_feather(file_path)
    splits = pd.read_feather(split_path)
    ids = splits.loc[splits["split"].isin(["train", "val"]), '_id']
    
    df = data[data['_id'].isin(ids)].copy()
    
    num_docs_to_modify = int(len(df) * (noise_percentage / 100))
    
    docs_to_modify = random.sample(df.index.tolist(), num_docs_to_modify)
    
    # Generate a pool of random words from the NLTK dictionary
    random_word_pool = words.words()
    
    noisy_documents = []
    for idx, doc in df.iterrows():
        if idx in docs_to_modify:
            words_in_doc = doc['text'].split()  
            num_words_to_substitute = int(len(words_in_doc) * 0.9)
            
            indices_to_substitute = random.sample(range(len(words_in_doc)), num_words_to_substitute)
            
            for word_idx in indices_to_substitute:
                words_in_doc[word_idx] = random.choice(random_word_pool)
            
            noisy_documents.append(' '.join(words_in_doc))
        else:
            noisy_documents.append(doc['text'])
    
    df['text'] = noisy_documents
    
    # Merge noisy df with original df
    combined_df = data.copy()  
    combined_df.update(df)     

    combined_df.to_feather(output_file_path)
    print(f"Noisy file saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input Feather file.")
    parser.add_argument("--split_path", type=str, required=True, help="Path to the split file with IDs.")
    parser.add_argument("--noise_percentage", type=float, required=True, help="Percentage of documents to modify (0-100).")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the modified Feather file.")
    args = parser.parse_args()

    add_noise_to_feather(args.file_path, args.split_path, args.noise_percentage, args.output_file_path)
