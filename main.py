import numpy as np
from src.feature_selection import forward_selection

def main():
    print("Welcome to Bertie Wooster's Feature Selection Algorithm.\n")
    filename = input("Type in the name of the file to test : ").strip() # User input for dataset file name

    filepath = f"data/{filename}"
    try:
        data = []
        with open(filepath, 'r') as file:
            for line in file:
                numbers = list(map(float, line.split())) # Splitting str into list of substr -> converting each substr to float via map
                data.append(numbers)  
        data_array = np.array(data)
        forward_selection(data_array)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return


if __name__ == "__main__":
    main()
