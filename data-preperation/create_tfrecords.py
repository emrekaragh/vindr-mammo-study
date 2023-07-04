import os
import argparse
import io
import numbers
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


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

def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def process_one_image(path: str, boxes: list, texts: list, labels: list, crop: bool):
    """
    Crops image if needed then convert image and its annotations to tf.train.Example
    :param path: path of image file
    :param boxes: list of bbox of objects in image related with each image
    :param texts: texts of objects related with each image
    :param labels: labels of objects related with each image
    :return: tf.train.Example Object
    """
    name = os.path.basename(path)
    image = Image.open(path)
    h, w = image.height, image.width
    xmin, ymin, xmax, ymax = np.split(np.array(boxes, dtype=np.float32), 4, -1)

    xmin, ymin, xmax, ymax = xmin / w, ymin / h, xmax / w, ymax / h

    xmin = np.clip(xmin, 0.0, 1.0).reshape(-1).tolist()
    ymin = np.clip(ymin, 0.0, 1.0).reshape(-1).tolist()
    xmax = np.clip(xmax, 0.0, 1.0).reshape(-1).tolist()
    ymax = np.clip(ymax, 0.0, 1.0).reshape(-1).tolist()

    image = image_to_byte_array(image)

    texts = [text.encode('utf-8') for text in texts]
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(h),
        'image/width': int64_feature(w),
        'image/filename': bytes_feature(name.encode('utf-8')),
        'image/source_id': bytes_feature(name.encode('utf-8')),
        'image/encoded': bytes_feature(image),
        'image/format': bytes_feature(b'jpg'),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(texts),
        'image/object/class/label': int64_list_feature(labels),
    }))


def visualize_tfrecords(record_path: str, number_of_records: int, number_of_test: int, save_dir: str):
    """
    Visualizes tf random instances from given tfrecord file
    :param record_path: tfrecord path
    :param number_of_records: number of total record in tfrecord file
    :param number_of_test: number of instances to be visualized
    :param save_dir: Path to the folder where the visualized images will be saved
    :return:
    """
    raw_image_dataset = tf.data.TFRecordDataset(record_path)

    parsed_images_dataset = raw_image_dataset.map(_parse_image_function)

    test_indexes = random.sample(list(range(number_of_records)), number_of_test)
    for counter, parsed_image_dataset in enumerate(parsed_images_dataset):
        if counter in test_indexes:
            image_bytes = parsed_image_dataset['image/encoded']
            name = parsed_image_dataset['image/filename'].numpy()
            width = parsed_image_dataset['image/width'].numpy()
            height = parsed_image_dataset['image/height'].numpy()
            xmin = parsed_image_dataset['image/object/bbox/xmin'].numpy()
            ymin = parsed_image_dataset['image/object/bbox/ymin'].numpy()
            xmax = parsed_image_dataset['image/object/bbox/xmax'].numpy()
            ymax = parsed_image_dataset['image/object/bbox/ymax'].numpy()
            label = parsed_image_dataset['image/object/class/text'].numpy()
            name_last = name.decode("utf-8")

            image = Image.fromarray(tf.image.decode_image(image_bytes).numpy())
            if xmin.any():
                coords = list(zip(xmin, ymin, xmax, ymax))
                draw = ImageDraw.Draw(image)
                for xmin, ymin, xmax, ymax in coords:
                    xmin *= width
                    xmax *= width
                    ymin *= height
                    ymax *= height
                    draw.rectangle(xy=[xmin, ymin, xmax, ymax], outline="red")

            save_path = os.path.join(save_dir, name_last)
            image.save(save_path)
        else:
            continue


def write_tfrecord(image_paths, annotation_paths, save_dir, crop):
    """
    Takes image and annotation paths, then writes them into tfrecord file
    :param image_paths: paths of image files
    :param annotation_paths:  paths of annotation files
    :param save_dir: Path to the file where record will be saved
    :param crop: Center crop will applied if crop was True
    :return:
    """
    text_to_label = {
        'person': 1,
        'Human': 1
    }
    images_and_annotations = list(zip(image_paths, annotation_paths))

    broken_images = []
    broken_annotations = []
    empty_annotations = []

    writer = tf.io.TFRecordWriter(save_dir)
    for img_path, annot_path in images_and_annotations:
        valid_flag = True
        try:
            width, height, boxes, texts = parse_one_xml(annot_path)
            if len(boxes) == 0:
                empty_annotations.append(os.path.basename(annot_path))
                valid_flag = False
        except:
            broken_annotations.append(os.path.basename(annot_path))
            valid_flag = False

        try:
            labels = [text_to_label[text] for text in texts]
            processed_image = process_one_image(img_path, boxes, texts, labels, crop).SerializeToString()
        except:
            broken_images.append(os.path.basename(img_path))
            valid_flag = False

        if valid_flag:
            writer.write(processed_image)
    writer.close()

    return broken_images, broken_annotations, empty_annotations


def main(images_dir: Path, ouput_dir: Path, annotations_dir: Path, visualize):
    
    os.makedirs(output_dir, exist_ok=True)

    
    
    return 
    # write train tfrecord
    train_record_save_dir = os.path.join(output_dir, 'train.record')
    train_broken_images, train_broken_annotations, train_empty_annotations = write_tfrecord(
        train_image_paths, train_annotation_paths, train_record_save_dir, crop
    )

    # write validation tfrecord
    validation_record_save_dir = os.path.join(output_dir, 'validation.record')
    validation_broken_images, validation_broken_annotations, validation_empty_annotations = write_tfrecord(
        validation_image_paths, validation_annotation_paths, validation_record_save_dir, crop
    )

    # visualize generated tfrecords for manuel testing
    train_visualize_dir = os.path.join(output_dir, 'visualization', 'train')
    os.makedirs(train_visualize_dir)
    validation_visualize_dir = os.path.join(output_dir, 'visualization', 'validation')
    os.makedirs(validation_visualize_dir)
    visualize_tfrecords(train_record_save_dir, len(train_image_paths), visualize, train_visualize_dir)
    visualize_tfrecords(validation_record_save_dir, len(validation_image_paths), visualize, validation_visualize_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts xml annotations and images to tfrecord file')
    parser.add_argument('-a', '--annotations-csv', required=True, type=str,
                        help='Path to csv file which contains annotations')
    parser.add_argument('-i', '--images-dir', required=True, type=str,
                        help='Folder path of image files')
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='Path to outputs to be saved. Needs to be folder, not file path')
    parser.add_argument('-v', '--visualize', required=False, type=int, default=50,
                        help='Number of images to be visualized. default:50')

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    annotations_csv_filepath = Path(args.annotations_csv)

    if not images_dir.is_file:
        raise NotADirectoryError(images_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    if not annotations_csv_filepath.is_file:
        raise FileNotFoundError(annotations_csv_filepath)

    main(input_dir, output_dir, csv_filepath, args.overwrite, args.num_workers)