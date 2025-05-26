import os
from pathlib import Path

def extract_prompts(input_folder, output_file):
    """
    Extract user prompts from multiple text files and combine them into a single file.
    
    Args:
        input_folder (str): Path to the folder containing input text files
        output_file (str): Path to the output file where prompts will be saved
    """
    # Get all .txt files in the input folder
    input_files = list(Path(input_folder).glob('*.txt'))
    
    # Store all prompts
    all_prompts = []
    
    # Process each file
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # Extract prompts (lines starting with "User: ")
            prompts = [line.replace('User: ', '').strip() 
                      for line in lines 
                      if line.strip().startswith('User: ')]
            
            all_prompts.extend(prompts)
    
    # Write all prompts to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for prompt in all_prompts:
            outfile.write(prompt + '\n')
    
    print(f"Processed {len(input_files)} files")
    print(f"Extracted {len(all_prompts)} prompts")
    print(f"Results saved to {output_file}")

# Example usage
input_folder = "/home/reza/TalkToModel/query_matching/prompts/dynamic"  # Replace with your input folder path
output_file = "combined_prompts.txt"        # Replace with your desired output file path

extract_prompts(input_folder, output_file)