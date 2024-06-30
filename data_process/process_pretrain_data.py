import time
import argparse

def find_longest_cds(mrna_sequence, start_codon='ATG', stop_codons=['TAG', 'TAA', 'TGA']):
    start_index = mrna_sequence.find(start_codon)
    longest_cds_info = None

    while start_index != -1:
        end_index = start_index + len(start_codon)
        while end_index < len(mrna_sequence):
            codon = mrna_sequence[end_index:end_index + 3]
            if codon in stop_codons and (end_index - start_index) % 3 == 0:
                current_cds_length = end_index - start_index + 3
                if longest_cds_info is None or current_cds_length > longest_cds_info['length']:
                    longest_cds_info = {
                        "CDS": mrna_sequence[start_index:end_index + 3],
                        "Start Index": start_index,
                        "End Index": end_index + 2,
                        "length": current_cds_length
                    }
                break
            else:
                end_index += 1
        start_index = mrna_sequence.find(start_codon, start_index + 1)
    return longest_cds_info

def mark_cds_in_sequence(sequence, cds_info):
    if cds_info:
        marked_sequence = list(sequence)
        start_index = cds_info['Start Index']
        end_index = cds_info['End Index']
        marked_sequence[start_index:end_index + 1] = ['['] + marked_sequence[start_index:end_index + 1] + [']']
        return ''.join(marked_sequence)
    else:
        return sequence

def split_sequence(sequence):
    result = []
    cds_flag = False
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
    return result

def process_fasta_and_split_sequence(input_file_path, output_file_path):
    start_time = time.time()  # Record start time

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        sequences = []
        current_sequence = ""
        for line in input_file:
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line.strip()
        if current_sequence:
            sequences.append(current_sequence)

        for mrna_sequence in sequences:
            longest_cds_info = find_longest_cds(mrna_sequence)
            marked_sequence = mark_cds_in_sequence(mrna_sequence, longest_cds_info)
            spaced_sequence = " ".join(split_sequence(marked_sequence))
            output_file.write(spaced_sequence + "\n")

    end_time = time.time()  # Record end time
    print(f"Process completed. Total runtime: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process FASTA file to find and mark the longest CDS, then split the sequence.')
    parser.add_argument('--input_file', type=str, default="data_process/pre-train/pre_input.fasta", help='Path to the input FASTA file.')
    parser.add_argument('--output_file', type=str, default="sample_data/pre.txt", help='Path to the output file.')

    args = parser.parse_args()

    process_fasta_and_split_sequence(args.input_file, args.output_file)
    print("Process completed. Results have been saved to:", args.output_file)