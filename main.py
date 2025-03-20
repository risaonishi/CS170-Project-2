import numpy as np
import math

def forward_selection(data):
    current_set_of_features = []  
    for i in range(1, data.shape[1]):  # Loop through each level of the search tree
        print(f"On the {i}th level of the search tree") 
        feature_to_add_at_this_level = [] 
        best_so_far_accuracy = 0  

        for k in range(1, data.shape[1]):  # Iterate through all features
            if k not in current_set_of_features:  # Check if the feature is not already selected
                print(f"-- Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)  # Evaluate accuracy
                if accuracy > best_so_far_accuracy:  # Update the best feature if accuracy improves
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)  # Add the best feature to the set
        
        print(f"On level {i} I added feature {feature_to_add_at_this_level} to current set")


def leave_one_out_cross_validation(data, current_set, feature_to_add):
    for i in range(1, data.shape[1]):  # Loop through each object in the dataset
        object_to_classify = data[i]  # The object we are trying to classify
        label_object_to_classify = data[i][0]  # The true label of the object

        nearest_neighbor_distance = float('inf')  
        nearest_neighbor_location = float('inf')  
        for k in range(1, data.shape[1]):
            print(f"Ask if {i} is nearest neighbor with {k}")
            if k != i:  # Skip the object we are trying to classify
                squared_differences = [(data[k][j] - object_to_classify[j])**2 for j in range(1, data.shape[1])]
                distance = math.sqrt(sum(squared_differences))  # Compute the Euclidean distance
                if distance < nearest_neighbor_distance:  # Update nearest neighbor if distance is smaller
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        print(f"The nearest neighbor to {i} is {nearest_neighbor_location} and its label is {nearest_neighbor_label}")



def main():
    print("Welcome to Risa Onishi's Feature Selection Algorithm.\n")
    filename = input("Type in the name of the file to test : ").strip() # User input for dataset file name

    filepath = f"data/{filename}"
    try:
        data = []
        with open(filepath, 'r') as file:
            for line in file:
                numbers = list(map(float, line.split())) # Splitting str into list of substr -> converting each substr to float via map
                data.append(numbers)  
        data_array = np.array(data)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    algorithm = input("Type the number of the algorithm you want to run.\n"
                      "\t1) Forward Selection\n"
                      "\t2) Backward Elimination\n")
    if algorithm == '1':
        forward_selection(data_array)
    # elif algorithm == '2':
        # backward_elimination(data_array)
    # else:
    #     print("Invalid input, please type 1 or 2.")
    #     return
    
    print(f"This dataset has {data_array.shape[1]-1} features (not including the class attribute), with {data_array.shape[0]} instances. ")
if __name__ == "__main__":
    main()
