import os

directory = input("Enter path to data directory: ")
separator = input("Enter the separator your files use: (Eg. my files use _ (up_0.csv)): ")
gestures = {}
for root, _, files in os.walk(directory):
    for file in files:
        file_name_split = file.split(separator)
        
        if file_name_split[0] in gestures:
            gestures[file_name_split[0]] = gestures[file_name_split[0]] + 1
        else:
            gestures[file_name_split[0]] = 1
print(gestures)

