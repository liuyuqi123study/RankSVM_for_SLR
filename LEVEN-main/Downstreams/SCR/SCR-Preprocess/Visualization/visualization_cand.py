import os
import json
import matplotlib.pyplot as plt
import jsonlines
def get_string_lengths_from_json(folder_path):
    """
    Traverse through the folder to read JSON files and collect string lengths.
    
    Args:
        folder_path (str): The path to the folder containing JSON files.
        
    Returns:
        dict: A dictionary where keys are file paths, and values are lists of string lengths.
    """
    lengths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        lengths.append(len(data['ajjbqk']))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
    
    return lengths

def get_length_from_a_json_file(path):
    # Load JSON data
    data=[]
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    # Specify the key whose value length you want
    target_key = 'q'

    # Extract lengths of the specific item
    lengths = []
    for item in data:
        if target_key in item and isinstance(item[target_key], str):  # Check if the key exists and is a string
            lengths.append(len(item[target_key]))
        else:
            lengths.append(None)  # Handle missing keys or non-string values

    # Print results
    #for idx, length in enumerate(lengths, start=1):
        #print(f"Dict {idx}: Length of '{target_key}' = {length}")
    return lengths

def visualize_string_lengths(string_lengths):
    """
    Visualize string lengths using a bar plot.
    
    Args:
        string_lengths (dict): A dictionary where keys are file paths, and values are lists of string lengths.
    """
    
    
    plt.figure(figsize=(10, 6))
    plt.hist(string_lengths,bins=30,color='skyblue',edgecolor='black',rwidth=0.8,density=True)
    plt.xlim(0,20000)
    #plt.xticks(ticks=np.arange(0,20000,1000),rotation=45)
    plt.xlabel("Length of Documents")
    plt.ylabel("Number")
    plt.title("Visualization of Candidates Lengths in LeCaRDv1")
    plt.tight_layout()
    plt.savefig('histogram_v1_candidates.pdf', format='pdf')

def visualize_string_lengths_query(string_lengths):
    plt.figure(figsize=(10, 6))
    plt.hist(string_lengths,bins=30,color='skyblue',edgecolor='black',rwidth=0.8,density=True)
    plt.xlim(0,1000)
    #plt.xticks(ticks=np.arange(0,20000,1000),rotation=45)
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
    plt.xlabel("Length of Documents")
    plt.ylabel("Percentage")
    plt.title("Visualization of Queries Lengths in LeCaRDv1")
    plt.tight_layout()
    plt.savefig('histogram_v1_queries.pdf', format='pdf')

# Main execution
if __name__ == "__main__":
    folder_path = "LEVEN-main/Downstreams/SCR/SCR-Experiment/input_data/candidates"  # Replace with the path to your folder
    string_lengths = get_string_lengths_from_json(folder_path)
    
    if string_lengths:
        visualize_string_lengths(string_lengths)
    else:
        print("No JSON files found or no string items in the JSON files.")

    #Visualize Lengths of the query file
    path='LEVEN-main/Downstreams/SCR/SCR-Preprocess/input_data/data/query/query.json'
    lengths=get_length_from_a_json_file(path)

    if lengths:
        visualize_string_lengths_query(lengths)
    else:
        print("No JSON files found or no string items in the JSON files.")

    
