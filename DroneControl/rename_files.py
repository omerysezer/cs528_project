import os

directory = input("Enter path to data directory: ")
suffix = input("Enter your initials: ")
separator = input("Enter the separator your files use: (Eg. my files use _ (up_0.csv)): ")

for root, _, files in os.walk(directory):
    for file in files:
        file_name_split = file.split(separator)
        file_name_split.insert(1, suffix)
        new_file_name = '_'.join(file_name_split)

        basename, extension = os.path.splitext(new_file_name)
        new_file_name = basename + '.csv'
        os.rename(os.path.join(root, file), os.path.join(root, new_file_name))
