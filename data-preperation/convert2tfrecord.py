from pathlib import Path
from argparse import ArgumentParser
import random
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_to_byte_array(image: Image) -> bytes:
    # Taken from https://stackoverflow.com/a/56055505
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def main(images_dir: Path, annotations_csv_filepath: Path, output_dir: Path, overwrite: bool):
    



if __name__ == '__main__':
    parser = ArgumentParser('Convert Vindr-Mammo dataset to TfRecord CLI')
    parser.add_argument('-i', '--images-dir', type=str, help='Path to where png images are located')
    parser.add_argument('-a', '-annotations-csv-filepath', type=str, help='Path to where findings csv file is located')
    parser.add_argument('-o', '--output-dir', type=str, help='Path to where output tfrecords and label_map to be saved')
    parser.add_argument('--overwrite', action='store_true', help='Pass this argument if you want overwrite outputs if they already exists')
    args = parser.parse_args()

    images_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    annotations_csv_filepath = Path(args.annotations_csv_filepath)
    overwrite = args.overwrite

    if not images_dir.is_dir:
        raise NotADirectoryError(images_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    if not annotations_csv_filepath.is_file:
        raise FileNotFoundError(annotations_csv_filepath)
    
    main(images_dir, annotations_csv_filepath, output_dir, overwrite)