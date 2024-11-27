import os
import hashlib


def hash_line(line):
    """Returns a hash for the given line."""
    return hashlib.md5(line.encode('utf-8')).hexdigest()


def find_duplicates_in_folder(folder_path):
    unique_lines = set()
    duplicate_lines = set()
    total_data_size = 0
    unique_data_size = 0

    # Loop through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        total_data_size += len(line.encode('utf-8'))
                        line_hash = hash_line(line.strip())
                        if line_hash in unique_lines:
                            duplicate_lines.add(line_hash)
                        else:
                            unique_lines.add(line_hash)
                            unique_data_size += len(line.encode('utf-8'))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    print(f"Total data size: {total_data_size} bytes")
    print(f"Unique data size: {unique_data_size} bytes")
    print(f"Duplicate data size: {total_data_size - unique_data_size} bytes")
    print(f"Number of duplicate lines: {len(duplicate_lines)}")
    print(f"Number of unique lines: {len(unique_lines)}")
    print(f"Percentage of unique data: {(unique_data_size / total_data_size) * 100} %")


if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    if os.path.isdir(folder_path):
        find_duplicates_in_folder(folder_path)
    else:
        print("Invalid folder path.")
