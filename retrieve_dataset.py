"""
retrieve_dataset.py

This script downloads and prepares the Freesound Loop Dataset from Zenodo.

This script performs the following tasks:
- Downloads all files from a specified Zenodo record ID
- Saves files to a target directory
- Automatically extracts ZIP archives into subdirectories

Usage:
    python retrieve_dataset.py --output_dir path/to/save

Dependencies:
- Python 3.10+
- requests
- os
- zipfile
"""
import requests
import os
import zipfile

def download_zenodo_record(record_id, save_path="."):
    """
    Downloads all files from a given Zenodo record ID and extracts zip files if present.

    Parameters:
        record_id (str): The Zenodo record ID.
        save_path (str): Directory where files should be saved. Defaults to current directory.
    """
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)

    if response.status_code == 200:
        files = response.json().get('files', [])
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            download_url = file['links']['self']
            filename = file['key']
            file_path = os.path.join(save_path, filename)

            print(f"Downloading {filename}...")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Saved {filename} to {file_path}")

            # Check if the downloaded file is a zip and extract it
            if filename.lower().endswith(".zip"):
                extract_path = os.path.join(save_path, os.path.splitext(filename)[0])
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extracted {filename} to {extract_path}")
    else:
        print("Failed to retrieve Zenodo record.")


if __name__ == "__main__":
    ### It might take some time (10-20min)
    download_zenodo_record("3967852")