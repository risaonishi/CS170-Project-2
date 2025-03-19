import numpy as np
import random

def forward_selection(data):
    current_set_of_features = []  # Initialize an empty set
    for i in range(1, data.shape[1]):
        print(f"On the {i}th level of the search tree") 
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0

        for k in range(1, data.shape[1]):
            if k not in current_set_of_features:
                print(f"-- Considering adding the {k} feature")
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        current_set_of_features.append(feature_to_add_at_this_level)
        
        print("On level", i, "I added feature", feature_to_add_at_this_level, "to current set")


def leave_one_out_cross_validation(data, current_set, feature_to_add):
    for i in range(1, data.shape[1]):
        object_to_classify = data[i]
        label_object_to_classify = data[i][0]
    return random.random()
