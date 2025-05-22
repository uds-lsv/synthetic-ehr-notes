import pandas as pd
import re
import argparse

def remove_instructions(df, column='text', primary_keyword='discharge condition', secondary_keyword='discharge instruction', third_keyword='discharge instructions'):
    # remove dicharge instructions and condition, since it is only repetition of what is stated in the medical note before
    def truncate(text):
        primary_pos = text.lower().find(primary_keyword)
        if primary_pos != -1:
            return text[:primary_pos].strip()
        secondary_pos = text.lower().find(secondary_keyword)
        if secondary_pos != -1:
            return text[:secondary_pos].strip()
        third_pos = text.lower().find(third_keyword)
        if third_pos != -1:
            return text[:third_pos].strip()
        return text.strip()
    df[column] = df[column].apply(truncate)
    return df

def remove_beginning(df, column='text', primary_keyword='chief complaint', secondary_keyword='major surgical or invasive procedure', third_keyword='history of present illness'):
    # remove begiining of note (admission data, discharge date, date of birth) since irrelevant and pseudonomized
    def truncate(text):
        primary_pos = text.lower().find(primary_keyword)
        if primary_pos != -1:
            return text[primary_pos:].lstrip()
        secondary_pos = text.lower().find(secondary_keyword)
        if secondary_pos != -1:
            return text[secondary_pos:].lstrip()
        third_pos = text.lower().find(third_keyword)
        if third_pos != -1:
            return text[:third_pos].lstrip()
        return text.strip()
    df[column] = df[column].apply(truncate)
    return df

def remove_medication_lines(df, column='text'):
    # remove medications since structured, not integrated in free text
    def is_medication_line(line):
        return re.search(r'\bmedications?\s*(?:[^:]*:)?\s*$', line, re.IGNORECASE) is not None
    for index, row in df.iterrows():
        lines = row[column].splitlines()
        lines_to_keep = []
        skip_lines = False
        for line in lines:
            if not skip_lines:
                if is_medication_line(line):
                    skip_lines = True
                else:
                    lines_to_keep.append(line)
            else:
                if line.strip() == '':
                    skip_lines = False
        df.at[index, column] = '\n'.join(lines_to_keep)
    return df

def remove_lines_starting_with_time(df, column='text'):
    # remove vitals since they contain to many digits
    def starts_with_time(line):
        return re.match(r'^___\s*\d{2}:\d{2}.*$', line.strip()) is not None
    for index, row in df.iterrows():
        lines = row[column].splitlines()
        lines_to_keep = []
        skip_lines = False
        for line in lines:
            if not skip_lines:
                if starts_with_time(line):
                    skip_lines = True
                else:
                    lines_to_keep.append(line)
            else:
                if line.strip() == '':
                    skip_lines = False
        df.at[index, column] = '\n'.join(lines_to_keep)
    return df

def clean_labs(text):
    # remove lab results since structured and too many digits
    pattern = re.compile(r'^(.*?(pertinent result|admission lab|discharge lab).*?)$', re.IGNORECASE | re.MULTILINE)
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def clean_text_column(df, column='text'):
    #remove facility since pseudomonized and irrelevant and empty lines
    def clean_text(text):
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not (line.strip().lower() == 'facility:' or line.strip() == '___')]
        return '\n'.join(cleaned_lines)
    df[column] = df[column].apply(clean_text)
    return df

def remove_line_breaks(text):
    # remove line breaks within a section
    lines = text.splitlines()
    preserved_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isupper() or line.endswith(':'):
            preserved_lines.append('\n' + line if i > 0 else line)
        else:
            preserved_lines.append(' ' + line if i > 0 else line)
        i += 1
    result = ''.join(preserved_lines)
    return result

def main(input_file, output_file):
    # load the Feather file
    df = pd.read_feather(input_file)

    # process the 'text' column
    df = remove_instructions(df, column='text')
    df = remove_beginning(df, column='text')
    df = remove_medication_lines(df, column='text')
    df = remove_lines_starting_with_time(df, column='text')
    df['text'] = df['text'].apply(clean_labs)
    df = clean_text_column(df, column='text')
    df['text'] = df['text'].apply(remove_line_breaks)

    # save the processed DataFrame to the output Feather file
    df.to_feather(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input Feather file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output Feather file.")
    args = parser.parse_args()
    main(args.input_file, args.output_file)