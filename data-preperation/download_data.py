import argparse
from pathlib import Path
import sys
import subprocess

import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from PIL import Image

def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    taken from: https://github.com/vinbigdata-medical/vindr-mammo/blob/master/visualize.py

    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    if ('WindowCenter' not in data) or\
       ('WindowWidth' not in data) or\
       ('PhotometricInterpretation' not in data) or\
       ('RescaleSlope' not in data) or\
       ('PresentationIntentType' not in data) or\
       ('RescaleIntercept' not in data):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element('PresentationIntentType').value
    if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
        print(f"{dicom_file} got processing file")
        return


    c = data.data_element('WindowCenter').value # data[0x0028, 0x1050].value
    w = data.data_element('WindowWidth').value  # data[0x0028, 0x1051].value
    if type(c)==pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    try:
        a = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get get pixel_array!')
        return

    slope = data.data_element('RescaleSlope').value
    intercept = data.data_element('RescaleIntercept').value
    a = a * slope + intercept

    try:
        pad_val = data.get('PixelPaddingValue')
        pad_limit = data.get('PixelPaddingRangeLimit', -99999)
        if pad_limit == -99999:
            mask_pad = (a==pad_val)
        else:
            if str(photometricInterpretation) == 'MONOCHROME2':
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f'{dicom_file} has no PixelPaddingValue')
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(f'{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}')
        except:
            print(f'{dicom_file} most frequent pixel value {sorted_pixels[0]}')

    # apply window
    mm = c - 0.5 - (w-1)/2
    MM = c - 0.5 + (w-1)/2
    a[a<mm] = 0
    a[a>MM] = 255
    mask = (a>=mm) & (a<=MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w-1) + 0.5) * 255

    if str( photometricInterpretation ) == 'MONOCHROME1':
        a = 255 - a

    a[mask_pad] = 0
    return a

def save_png(input_array: np.ndarray, target_file_path: Path):
    image = input_array.astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(str(target_file_path) )

def check_file_integrity(file_path, expected_hash):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()

    return file_hash == expected_hash, file_hash

def wget_download(url: str, filepath: Path, output_directory: str, username: str, password: str, sha256: str = None, max_attempts=10):
    #print('download_one_file: \turl: {}, filepath: {}, output_directory: {}, username: {}, password: {}, sha256: {}'.format(url, filepath, output_directory, username, password, sha256))
    checksum = True if sha256 else False

    for i in range(max_attempts):
        # Build the wget command with the desired options
        wget_command = f"wget -r -N -c -np --no-directories -P {output_directory} --user {username} --password {password} {url}"

        #input(':')

        # Use subprocess to execute the command
        subprocess.run(wget_command, shell=True)

        # Validate the file
        if filepath.exists():
            if checksum:
                if check_file_integrity(file_path=filepath, expected_hash=sha256):
                    return True
            else:
                return True
    return False

def remove_file(): pass

def unzip_one_file():
    pass




"""
def download_files(base_url: str, df: pd.DataFrame, output_dir: Path, username: str, password: str):    
    print('\n', '-'*9, 'DOWNLOADING THE FILES', '-'*9, '\n')
    missing_files = []
    for index, row in df.iterrows():
        print(row, type(row))
        url = '{}/{}'.format(base_url, row['relative_path'])
        filepath = output_dir.joinpath(row['relative_path'].split('/')[2])
        is_downladed = download_one_file(url, filepath, output_dir, username, password, row['sha256_hash'])
        if not is_downladed:
            missing_files.append(row['image_id'])

    return missing_files
"""

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

    missing_files = download_files(base_url, df_files, output_dir, username, password)
    
    print('\n\n', '-'*9, 'FILES THAT CANNOT BE DOWNLOADED', '-'*9, '\n')
    print('Total:', len(missing_files), '\n')
    for file in missing_files:
        print(file)
    
