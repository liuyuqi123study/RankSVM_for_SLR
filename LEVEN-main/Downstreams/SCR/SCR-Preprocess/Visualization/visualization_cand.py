import os
import json
import matplotlib.pyplot as plt

def get_string_lengths_from_json(folder_path):
    """
    Traverse through the folder to read JSON files and collect string lengths.
    
    Args:
        folder_path (str): The path to the folder containing JSON files.
        
    Returns:
        dict: A dictionary where keys are file paths, and values are lists of string lengths.
    """
    string_lengths = {}
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        lengths = []
                        for key, value in data.items():
                            if isinstance(value, str):
                                lengths.append(len(value))
                        string_lengths[file_path] = lengths
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
    
    return string_lengths

def visualize_string_lengths(string_lengths):
    """
    Visualize string lengths using a bar plot.
    
    Args:
        string_lengths (dict): A dictionary where keys are file paths, and values are lists of string lengths.
    """
    file_names = list(string_lengths.keys())
    lengths = [sum(lengths) for lengths in string_lengths.values()]
    
    plt.figure(figsize=(10, 6))
    plt.barh(file_names, lengths, color='skyblue')
    plt.xlabel("Total Length of Strings")
    plt.ylabel("File Paths")
    plt.title("Visualization of String Lengths in JSON Files")
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    folder_path = "path/to/your/folder"  # Replace with the path to your folder
    string_lengths = get_string_lengths_from_json(folder_path)
    
    if string_lengths:
        visualize_string_lengths(string_lengths)
    else:
        print("No JSON files found or no string items in the JSON files.")
