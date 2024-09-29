import os
import shutil

def clear_directory(output_dir):
    # Check if the directory contains files or folders
    if os.listdir(output_dir):
        # Loop through the files and delete each one
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"All files in '{output_dir}' have been deleted.")
    else:
        print(f"'{output_dir}' is already empty.")

def main():
    # clear_directory('D:\WorkSpace\WiFi-Vision-Counting\data-collection\cv_main\session_1')
    pass

if __name__ == "__main__":
    main()