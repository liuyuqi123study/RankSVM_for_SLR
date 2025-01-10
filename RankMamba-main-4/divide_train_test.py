import json
import numpy as np
from sklearn.model_selection import KFold

# Step 1: Load the JSON data
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Step 2: Create 5 folds
def create_folds(data, n_splits=5, random_state=42):
    # Convert data to a numpy array if it's not already
    data = np.array(data)
    
    # Shuffle data
    np.random.seed(random_state)
    np.random.shuffle(data)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    folds = []
    for train_index, test_index in kf.split(data):
        train_fold = data[train_index]
        test_fold = data[test_index]
        folds.append((train_fold.tolist(), test_fold.tolist()))
    
    return folds

# Step 3: Save each fold into separate JSON files
def save_folds(folds, base_filename):
    for i, (train_data, test_data) in enumerate(folds):
        train_file = f'{base_filename}_fold_{i+1}_train.json'
        test_file = f'{base_filename}_fold_{i+1}_test.json'
        with open(train_file, 'w') as file:
            json.dump(train_data, file, indent=4)
        with open(test_file, 'w') as file:
            json.dump(test_data, file, indent=4)

# Main function to execute the above steps
def main(input_file, base_filename, n_splits=5):
    # Load the data
    data = load_json(input_file)
    
    # Create folds
    folds = create_folds(data, n_splits)
    
    # Save the folds
    save_folds(folds, base_filename)

# Example usage
if __name__ == "__main__":
    input_file = ''         # Your input JSON file
    base_filename = 'fold'           # Base filename for output JSON files
    main(input_file, base_filename)
