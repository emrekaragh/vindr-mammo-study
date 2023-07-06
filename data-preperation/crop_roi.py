from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
from PIL import Image
import numpy as np

def find_largest_connected_component(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    # Find the index of the largest connected component (excluding background component)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a binary mask for the largest connected component
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == largest_component_index] = 255

    # Find the bounding box of the largest connected component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    xmin = x
    ymin = y
    xmax = xmin + w
    ymax = ymin + h
    
    return (xmin, ymin, xmax, ymax)

def breast_detector(image):
    return (xmin, ymin, xmax, ymax)

def crop_image(image, xmin, ymin, xmax, ymax):    
    # Crop image with given bbox
    return image[ymin:ymax, xmin:xmax]

def save_png(input_array: np.ndarray, target_file_path: Path):
    image = input_array.astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(str(target_file_path))

def process_one_image(image_path: Path, target_filepath: Path, method: str = 'BD', overwrite: bool = False):
    try:
        if overwrite or not target_filepath.exists():
            image = cv2.imread(str(image_path))
            if method == 'BD':
                # Find the Breast bbox coordinates by running detector model
                bbox_coordinates = breast_detector(image)
            elif method == 'LC':
                # Find the largest connected component's bbox coordinates
                bbox_coordinates = find_largest_connected_component(image)
            else:
                raise ValueError('Invalid method, method can be one of LC or BD')
            
            image_id = target_filepath.stem
            xmin, ymin, xmax, ymax = bbox_coordinates
            width = xmax-xmin
            height = ymax-ymin
            image_info.append((image_id, xmin, ymin, xmax, ymax, width, height))
            
            cropped = crop_image(image, xmin, ymin, xmax, ymax)
            # Save the cropped component as PNG image
            save_png(cropped, target_filepath)
    except Exception as e:
        print(image_path.name)
        print(e)

def thread_fn(args):
    image_path, target_filepath, method, overwrite = args
    process_one_image(image_path=image_path, target_filepath=target_filepath, method=method, overwrite=overwrite)

def get_png_file_paths(wd: Path):
    png_files = []

    # Iterate over all files and directories recursively
    for path in wd.glob('**/*'):
        if path.is_file() and path.suffix.lower() == '.png':
            png_files.append(path.resolve())

    return png_files

def main(input_dir: Path, output_dir: Path, method: str = 'BD', overwrite: bool = False, num_workers = None):
    
    png_filepaths = get_png_file_paths(input_dir)
    source_target_filepaths = []
    for path in png_filepaths:
        source_filepath: Path = path
        image_name = source_filepath.name
        target_filepath = output_dir.joinpath((image_name))
        source_target_filepaths.append((source_filepath, target_filepath))

    # Create a thread pool with a maximum of num_workers threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Prepare the arguments for each file to be processed
        arguments = [(source_filepath, target_filepath, method, overwrite) for (source_filepath, target_filepath) in source_target_filepaths]
        # Submit the tasks to the thread pool
        executor.map(thread_fn, arguments)

    with open(output_dir.joinpath('image_info.csv'), 'w') as f:
        f.write('study_id, image_id, roi_xmin, roi_ymin, roi_xmax, roi_ymax, roi_width, roi_height\n')
        for info in image_info:
            f.write(','.join(str(item) for item in info)+'\n')


if __name__ == '__main__':
    image_info = []
    parser = ArgumentParser('Crop ROI')
    parser.add_argument('-i', '--input-dir', required=True, type=str, help='')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output direcotory')
    parser.add_argument('-m', '--method', required=False, type=str, default='BD', help='Method to extract ROI, LC(Largest Component or BD(Breast Detector))')
    parser.add_argument('--overwrite', action='store_true', help='Pass this parameter to overwrite existing files')
    parser.add_argument('-n', '--num-workers', required=False, type=int, default=60, help='Pass this parameter to overwrite existing files')

    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.method not in ['LC', 'BD']:
        raise ValueError('Invalid method, method can be one of LC or BD')
    if not input_dir.is_file:
        raise NotADirectoryError(input_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    
    main(input_dir, output_dir, args.method, args.overwrite, args.num_workers)
