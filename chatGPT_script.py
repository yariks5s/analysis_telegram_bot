import os

def is_binary_file(file_path):
    """
    Check if a file is binary by reading its first few bytes.
    """
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)  # Read the first 1024 bytes
        if b'\0' in chunk:  # Binary files usually contain null bytes
            return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return True
    return False

def save_files_content_to_text(directory, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    if is_binary_file(file_path):
                        print(f"Skipping binary file: {filename}")
                        continue
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    output.write(f"{filename}:\n\n{content}\n\n\n")
        print(f"Content written to {output_file} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Directory containing files
input_directory = "/Users/yaroslav/cryptoBot/"

# Output file
output_filename = "output.txt"

save_files_content_to_text(input_directory, output_filename)