from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
import ast 

import cv2
import numpy as np
from PIL import Image
import pandas as pd

from emphasise_roi import EmphasiseEllipticalROI, Roi

class Bbox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
    
    def flip_h(self, width, height):
        xmin = width - self.xmax
        ymin = self.ymin
        xmax = width - self.xmin
        ymax = self.ymax
        return Bbox(xmin, ymin, xmax, ymax)
    
    def to_roi(self):
        return Roi(self.xmin, self.ymin, self.xmax, self.ymax)

    def get_tuple(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)
        
def horizontal_flip(image: np.ndarray):
    return cv2.flip(image, 1)

def transparency(image: np.ndarray, rois:List[Roi], alpha: float = 0.3, beta: float = 0.9):
    height, width = image.shape

    elliptical_kernel_generator = EmphasiseEllipticalROI(kernel_width=width, kernel_height=height, rois=rois)
    kernel = elliptical_kernel_generator(alpha=alpha, beta=beta)

    return np.multiply(image.astype(np.float32), kernel).astype(np.uint8)    

def save_png(input_array: np.ndarray, target_file_path: Path):
    image = input_array.astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(str(target_file_path))

def process_one_image(image_filepath: Path, output_dir: Path, birads: str, bboxes_str):
    alpha_beta_valeus = {'BI-RADS 3':[(0.4, 0.9), (0.3, 0.8)], 'BI-RADS 4':[(0.2, 0.75), (0.45, 0.95)], 'BI-RADS 5':[(0.1, 0.6), (0.2, 0.9), (0.3, 0.66), (0.35, 0.78), (0.4, 0.75), (0.45, 0.84) ,(0.5, 0.95), (0.6, 0.9)] }

    image = cv2.imread(str(image_filepath), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    image_id = image_filepath.stem
    bboxes = []
    for xmin, ymin, xmax, ymax  in ast.literal_eval(bboxes_str):
        bboxes.append(Bbox(xmin, ymin, xmax, ymax))

    images = {}
    images['original'] = {'image_id': image_id, 'array': image, 'bboxes': bboxes}

    images['h_flipped'] = {'image_id': (image_id + '_aug_hf'), 'array':horizontal_flip(image), 'bboxes': [bbox.flip_h(width, height) for bbox in bboxes]}

    augmented_images = {}
    for index, (alpha, beta) in enumerate(alpha_beta_valeus[birads]):
        for image in images.values():
            new_image_id = (image['image_id'] + '_tr_' + str(index+1))
            rois = [bbox.to_roi() for bbox in image['bboxes']]
            new_array = transparency(image=image['array'], rois=rois, alpha=alpha, beta=beta)
            augmented_images[new_image_id] = {'image_id': new_image_id, 'array': new_array, 'bboxes': image['bboxes']}

    images.update(augmented_images)
    images.pop('original')
    for image in images.values():
        save_png(image['array'], output_dir.joinpath(image['image_id']+'.png'))


def thread_fn(args):
    image_filepath, output_dir, birads, bboxes_str = args
    process_one_image(image_filepath, output_dir, birads, bboxes_str)

def main(input_dir: Path, output_dir: Path, csv_filepath: Path, overwrite: bool = False, num_workers=None):    
    df = pd.read_csv(csv_filepath)
    #condition = df['image_id'].isin(['01fb871dc222684a9950609b62b76772', '02d253f51556e2e0af63525de2e9ff74', '01df962b078e38500bf9dd9969a50083'])
    #df_filtered = df[condition]

    thread_args = []
    for index, row in df.iterrows():
        filepath = input_dir.joinpath((row['image_id']+'.png'))
        birads = row['finding_birads']
        bboxes_str = row['bboxes']
        thread_args.append((filepath,  output_dir, birads, bboxes_str))

    # Create a thread pool with a maximum of num_workers threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit the tasks to the thread pool
        executor.map(thread_fn, thread_args)


if __name__ == '__main__':    
    parser = ArgumentParser('Data Augmentation')
    parser.add_argument('-i', '--input-dir', required=True, type=str, help='Input directory that contain images')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output direcotory to images be saved into')
    parser.add_argument('--info-csv', required=True, type=str, help='CSV file that contains information about how augmentation to be applied')
    parser.add_argument('--overwrite', action='store_true', help='Pass this parameter to overwrite existing files')
    parser.add_argument('-n', '--num-workers', required=False, type=int, default=60,
                        help='Number of workers can run parallely')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_filepath = Path(args.info_csv)

    if not input_dir.is_file:
        raise NotADirectoryError(input_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    if not csv_filepath.is_file:
        raise FileNotFoundError(csv_filepath)

    main(input_dir, output_dir, csv_filepath, args.overwrite, args.num_workers)

