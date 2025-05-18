import csv
import os

relative_path = os.path.join('water')
r_folder_format = 'water_data.csv'
w_folder_format = 'households_{}'
w_file_format = os.path.join('households_{}', '{}.csv')
wd = os.getcwd()

FILE_ROWS = 40960  # element number within a section


# Read the CSV file
def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        # Return the header and the rows
        return [row for row in reader]

# Write the CSV file
def gen_csv(id):
    # Read the CSV file
    r_folder_path = os.path.join(wd, relative_path, r_folder_format.format(id))
    with open(r_folder_path, 'r') as file:
        lines = file.readlines()[1:-1]

    # Create a dictionary to store the data
    my_map = {}
    # Iterate through the rows
    for line in lines:
        # Replace the spaces with semicolons and the Null values with 0.001
        line = line.replace(' ', ';').replace('Null\r', '0.001')
        # Split the line by semicolons
        slices = line.split(';')
        # Get the first slice as the key
        file_key = slices[0]
        # If the key is not in the dictionary, add it
        if file_key not in my_map:
            # Add the key to the dictionary
            my_map[file_key] = []
        # Append the slices to the dictionary
        my_map[file_key].append(slices[1:])

    # Create a new folder to store the data
    w_folder_path = os.path.join(wd, relative_path, w_folder_format.format(FILE_ROWS))
    # Create the folder if it does not exist
    os.makedirs(w_folder_path, exist_ok=True)

    # Iterate through the dictionary
    for key, value in my_map.items():
        # If the length of the value is less than the file rows, continue
        if len(value) < FILE_ROWS:
            print(f"[{key}] has {len(value)} rows.")
            continue
        # Create a new file to store the data
        w_file_path = os.path.join(wd, relative_path, w_file_format.format(FILE_ROWS, key))
        # Write the data to the file
        with open(w_file_path, 'w', newline='') as new_file:
            writer = csv.writer(new_file)
            writer.writerows(value)

    print(f"len: {len(my_map)}")


def main():
    print(wd)
    gen_csv(1)


if __name__ == "__main__":
    main()