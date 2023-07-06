import os
from pathlib import Path
from argparse import ArgumentParser

from zipfile import ZipFile
import pydicom
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
    image_pil.save(str(target_file_path))


def extract_one_file(zip_filepath: Path, relative_path: str, output_dir: Path):
    with ZipFile(input_filepath, 'r') as zip_obj:
        zip_obj.extract(relative_path, path=output_dir)
    
def process_one_file(zip_filepath: Path, output_dir: Path, relative_path:str, overwrite: bool = False):
    target_filepath = output_dir.joinpath(relative_path)
    processed_filepath = target_filepath.with_suffix('.png') if str(target_filepath).endswith('.dicom') else target_filepath
    if overwrite or not processed_filepath.exists():
        extract_one_file(zip_filepath, relative_path, output_dir) 
        if target_filepath.suffix == '.dicom':
            image_array = convert_dicom_to_png(str(target_filepath))
            save_png(image_array, processed_filepath)
            os.remove(target_filepath)

def process_file(args):
    zip_filepath, output_dir, relative_path, overwrite = args
    process_one_file(zip_filepath=zip_filepath, output_dir=output_dir, relative_path=relative_path, overwrite=overwrite)

def main(input_filepath: Path, output_dir: Path, includes: Path = None, overwrite: bool = False, num_workers = None):
    os.makedirs(output_dir, exist_ok=True)

    if includes:
        included_files = []
        with open(includes, 'r') as file:
            for line in file.readlines():
                included_files.append(line.rstrip('\n'))

    with ZipFile(input_filepath, 'r') as zip_obj:
        relative_paths = [name for name in zip_obj.namelist()]

    filtered_relative_paths = [rel_path for rel_path in relative_paths if Path(rel_path).name in included_files]

    # Create a thread pool with a maximum of num_workers threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Prepare the arguments for each file to be processed
        arguments = [(input_filepath, output_dir, relative_path, overwrite) for relative_path in filtered_relative_paths]
        # Submit the tasks to the thread pool
        executor.map(process_file, arguments)

if __name__ == '__main__':
    parser = ArgumentParser('Unzip Zip Files (dicoms to be converted to png files)')
    parser.add_argument('-i', '--input-filepath', required=True, type=str, help='Zip file\'s path')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output direcotory')
    parser.add_argument('--includes', required=False, type=str, help='Pass to file that contains filenames to be included')
    parser.add_argument('--overwrite', action='store_true', help='Pass this parameter to overwrite existing files')
    parser.add_argument('-n', '--num-workers', required=False, type=int, default=120, help='Pass this parameter to overwrite existing files')

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    input_filepath = Path(args.input_filepath)
    includes = Path(args.includes)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    if not input_filepath.is_file:
        raise FileNotFoundError(input_filepath)
    if includes and not includes.is_file:
        raise FileNotFoundError(includes)
    
    main(input_filepath, output_dir, includes, args.overwrite, args.num_workers)
