import os
def ensure_dir_for_file(file_path):
    """Ensures the directory for a given file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
