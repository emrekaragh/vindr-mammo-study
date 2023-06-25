import hashlib
from pathlib import Path
import subprocess
import sys
import pandas as pd

def check_file_integrity(file_path, expected_hash):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    return file_hash == expected_hash, file_hash

def download_one_file(url: str, filepath: Path, output_directory: str, username: str, password: str, sha256: str = None, max_attempts=10):
    checksum = True if sha256 else False

    for i in range(max_attempts):
        # Build the wget command with the desired options
        wget_command = f"wget -r -N -c -np -P {output_directory} --user {username} --password {password} {url}"

        # Use subprocess to execute the command
        subprocess.run(wget_command, shell=True)

        # Validate the file
        if filepath.exists():
            if check_file_integrity(file_path=filepath, expected_hash=sha256):
                return True

    return False

def download_files(base_url: str, df: pd.DataFrame, output_dir: Path, username: str, password: str):    
    print('\n', '-'*9, 'DOWNLOADING THE FILES', '-'*9, '\n')
    files_not_downloaded = []
    df['is_downloaded'] = df.apply((lambda row: download_one_file(base_url, df['relative_path'], output_dir, username, password, df['sha256_hash'])))
    return df

if __name__ == '__main__':
    output_dir_str = sys.argv[1] 
    username = sys.argv[2]
    password = sys.argv[3]
    base_url = 'https://physionet.org/files/vindr-mammo/1.0.0'

    output_dir = Path(output_dir_str)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)

    pwd = Path(__file__).parent
    file_info_csv_path = pwd.joinpath('data', 'images_to_download.csv')
    df_files = pd.read_csv(file_info_csv_path)

    df_files_processed = download_files(base_url, df_files, output_dir, username, password)
    
    print('\n\n', '-'*9, 'FILES THAT CANNOT BE DOWNLOADED', '-'*9, '\n')
    missing_files = df_files_processed[df_files_processed['is_downloaded'] == False]['image_id'].tolist()
    print('Total:', len(missing_files), '\n')
    for file in missing_files:
        print(file)
    
