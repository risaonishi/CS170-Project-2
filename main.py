import numpy as np
import math
import time

def forward_selection(data):
    current_set_of_features = []  
    print("Beginning search.")
    best_overall_set = []
    best_overall_accuracy = 0
    
    for i in range(1, data.shape[1]):  # Loop through each level of the search tree
        feature_to_add_at_this_level = None 
        best_so_far_accuracy = 0  

        for k in range(1, data.shape[1]):  # Iterate through all features
            if k not in current_set_of_features:  # Check if the feature is not already selected
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, [k])  # Evaluate accuracy
                print(f"Using feature(s) {set(current_set_of_features + [k])} accuracy is {accuracy*100:.1f}%")
                
                if accuracy > best_so_far_accuracy:  # Update the best feature if accuracy improves
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)  # Add the best feature to the set
            print(f"Feature set {set(current_set_of_features)} was best, accuracy is {best_so_far_accuracy*100:.1f}%\n")
            
            if best_so_far_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_so_far_accuracy
                best_overall_set = list(current_set_of_features)
    
    print(f"Finished search!! The best feature subset is {set(best_overall_set)}, which has an accuracy of {best_overall_accuracy*100:.1f}%")

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    features_to_consider = current_set + feature_to_add  
    for i in range(data.shape[0]):  # Loop through each object in the dataset
        object_to_classify = data[i, features_to_consider]  # The object we are trying to classify
        label_object_to_classify = data[i][0]  # The true label of the object

        nearest_neighbor_distance = float('inf')  
        nearest_neighbor_location = float('inf')  
        for k in range(data.shape[0]): # Asking if i is nearest neighbor with k
            if k != i:  # Skip the object we are trying to classify
                distance = math.sqrt(np.sum((data[k,features_to_consider] - object_to_classify)**2))  # Calculate the Euclidean distance
                if distance < nearest_neighbor_distance:  # Update nearest neighbor if distance is smaller
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location,0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy

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
    print(f"\nThis dataset has {data_array.shape[1]-1} features (not including the class attribute), with {data_array.shape[0]} instances.")
    all_feature_accuracy = leave_one_out_cross_validation(data_array, list(range(1, data_array.shape[1])), [])
    print(f"\nRunning nearest neighbor with all {data_array.shape[1]-1} features, using \"leave-one-out\" evaluation, I get an accuracy of {all_feature_accuracy*100:.1f}%")
    if algorithm == '1':
        forward_selection(data_array)
    elif algorithm == '2':
        backward_elimination(data_array)
    else:
        print("Invalid input, please type 1 or 2.")
        return
    
if __name__ == "__main__":
    main()
