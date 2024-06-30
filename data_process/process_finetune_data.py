import argparse
import csv
import os
import glob

def split_sequence(sequence, option):
    """
    Split the sequence according to the specified option.

    Args:
        sequence (str): The sequence to be split.
        option (str): The splitting option, including 'utr', 'codon', and 'complete'.

    Returns:
        str: The split sequence, with segments separated by spaces.
    """
    result = []
    cds_flag = False  # Flag for being inside a CDS sequence
    cds_sequence = ""

    for char in sequence:
        if char == '[':
            cds_flag = True
            if cds_sequence:
                result.extend(list(cds_sequence))
                cds_sequence = ""
        elif char == ']':
            cds_flag = False
            if cds_sequence:
                cds_tokens = [cds_sequence[i:i+3] for i in range(0, len(cds_sequence), 3)]
                result.extend(cds_tokens)
                cds_sequence = ""
        elif cds_flag:
            cds_sequence += char
        else:
            result.append(char)

    if cds_sequence:  # Handle any trailing CDS sequence
        cds_tokens = [cds_sequence[i:i+3] for i in range(0, len(cds_sequence), 3)]
        result.extend(cds_tokens)

    if option == "utr":
        return " ".join(list(sequence))
    elif option == "codon":
        return " ".join([sequence[i:i+3] for i in range(0, len(sequence), 3)])
    elif option == "complete":
        return " ".join(result)

def process_csv(input_file, output_file, option):
    """
    Process sequences in a CSV file and split them according to the specified option.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        option (str): Splitting option.
    """
    # Ensure the output path exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)

    try:
        with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)

            for idx, row in enumerate(csv_reader):
                if not row:  # Skip empty rows
                    continue
                if idx == 0:
                    csv_writer.writerow(['sequence', 'label'])
                else:
                    processed_sequence = split_sequence(row[0], option)
                    csv_writer.writerow([processed_sequence] + row[1:])
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

def process_path(input_dir, output_dir, option):
    """
    Process all CSV files in the input path and output them to the specified path with the same filenames.

    Args:
        input_dir (str): Path to the input path containing CSV files.
        output_dir (str): Path to the output path where processed files will be stored.
        option (str): Splitting option.
    """
    # Ensure the output path exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each CSV file in the input path
    for csv_file in glob.glob(os.path.join(input_dir, '*.csv')):
        file_name = os.path.basename(csv_file)
        output_file = os.path.join(output_dir, file_name)
        process_csv(csv_file, output_file, option)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sequences in all CSV files in the specified path and output them to another path while maintaining filenames.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where processed files will be stored.")
    parser.add_argument("--split_option", choices=['utr', 'codon', 'complete'], default='codon',
                        help="Splitting option: 'utr' (single character split), 'codon' (triplet split), or 'complete' (mixed split). Default is 'codon'.")

    args = parser.parse_args()

    # Process each CSV file found in the input directory
    process_path(args.input_dir, args.output_dir, args.split_option)

    print("Splitting of all CSV files completed. Results have been saved to:", args.output_dir)
