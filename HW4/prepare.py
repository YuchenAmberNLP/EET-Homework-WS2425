import os
import shutil

# Define the base directory and source Enron folders
base_dir = '/Users/sifan/Library/Mobile Documents/com~apple~CloudDocs/MA/CL/EET/4'  # Replace with your actual base directory
enron_dirs = {
    'train': ['enron1', 'enron2', 'enron3', 'enron4'],
    'dev': ['enron5'],
    'test': ['enron6']
}

# Function to ensure target directories exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main processing function
def organize_enron_data(base_dir, enron_dirs):
    for target, sources in enron_dirs.items(): #{'train': ['enron1', 'enron2', 'enron3', 'enron4'], 'dev': ['enron5'], 'test': ['enron6']}
        # Create target spam and ham directories for each split (train/dev/test)
        target_spam_dir = os.path.join(base_dir, target, 'spam') # create a target spam directory for each split: train, dev, test
        target_ham_dir = os.path.join(base_dir, target, 'ham') # create a target ham directory for each split: train, dev, test
        ensure_dir(target_spam_dir) # ensure the target spam directory exists
        ensure_dir(target_ham_dir)

        for source in sources: # ['enron1', 'enron2', 'enron3', 'enron4'] for train, ['enron5'] for dev, ['enron6'] for test
            source_dir = os.path.join(base_dir, source) # create a source directory every source 

            # Define source spam and ham folders, split them into spam and ham
            source_spam_dir = os.path.join(source_dir, 'spam') 
            source_ham_dir = os.path.join(source_dir, 'ham')

            # Copy all files from source spam and ham to target spam and ham
            for file_name in os.listdir(source_spam_dir): # list all files in the source spam directory
                source_file = os.path.join(source_spam_dir, file_name)
                shutil.copy(source_file, target_spam_dir)

            for file_name in os.listdir(source_ham_dir):
                source_file = os.path.join(source_ham_dir, file_name)
                shutil.copy(source_file, target_ham_dir)

    print("Files have been organized successfully.")

# Run the function
organize_enron_data(base_dir, enron_dirs)
