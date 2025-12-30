from huggingface_hub import login
from huggingface_hub import snapshot_download
import zipfile
import os

login()
snapshot_download(repo_id="PNU-Infosec/aegis-assets",local_dir="./assets")


zip_file_path = './assets/datasets/active_dataset.zip'
extraction_path = './assets/extracted_datasets/'

os.makedirs(extraction_path, exist_ok=True)

try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
    print(f"Successfully unzipped '{zip_file_path}' to '{extraction_path}'")
except zipfile.BadZipFile:
    print(f"Error: '{zip_file_path}' is not a valid ZIP file.")
except FileNotFoundError:
    print(f"Error: ZIP file not found at '{zip_file_path}'.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")