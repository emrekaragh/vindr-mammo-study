import os
import argparse
import io
import numbers
import random
import ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import traceback

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
    image.save(imgByteArr, format='PNG')
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


def process_one_image(image_path: Path, boxes: list, texts: list, labels: list):
    """
    :param image_path: path of image file
    :param boxes: list of bbox of objects in image related with each image
    :param texts: texts of objects related with each image
    :param labels: labels of objects related with each image
    :return: tf.train.Example Object
    """
    name = image_path.name
    image = Image.open(image_path).convert('RGB')

    h, w = image.height, image.width
    xmin, ymin, xmax, ymax = boxes
    xmin = np.array(xmin, dtype=np.float32)
    ymin = np.array(ymin, dtype=np.float32)
    xmax = np.array(xmax, dtype=np.float32)
    ymax = np.array(ymax, dtype=np.float32)
    #xmin, ymin, xmax, ymax = np.split(np.array(boxes, dtype=np.float32), 4, -1)


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
        'image/format': bytes_feature(b'png'),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(texts),
        'image/object/class/label': int64_list_feature(labels),
    }))


def visualize_tfrecords(record_path: Path, save_dir: Path, number_of_validation: int):
    """
    Visualizes tf random instances from given tfrecord file
    :param record_path: tfrecord path
    :param save_dir: Path to the folder where the visualized images will be saved
    :param number_of_validation: number of instances to be visualized
    :return:
    """
    dataset = tf.data.TFRecordDataset(record_path)
    
    dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer_size according to your dataset size
    random_samples = []
    for record in dataset.take(number_of_validation):
        random_samples.append(record)
    
    parsed_images_dataset = []
    for sample in random_samples:
        parsed_images_dataset.append(_parse_image_function(sample))

    for counter, parsed_image in enumerate(parsed_images_dataset):
        #print(parsed_image)
        image_bytes = parsed_image['image/encoded']
        name = parsed_image['image/filename'].numpy()
        width = parsed_image['image/width'].numpy()
        height = parsed_image['image/height'].numpy()
        xmin = parsed_image['image/object/bbox/xmin'].numpy()
        ymin = parsed_image['image/object/bbox/ymin'].numpy()
        xmax = parsed_image['image/object/bbox/xmax'].numpy()
        ymax = parsed_image['image/object/bbox/ymax'].numpy()
        label = parsed_image['image/object/class/label'].numpy()
        text = parsed_image['image/object/class/text'].numpy()
        name_last = name.decode("utf-8")

        image = Image.fromarray(tf.image.decode_image(image_bytes).numpy())
        if xmin.any():
            coords = list(zip(xmin, ymin, xmax, ymax, label))
            draw = ImageDraw.Draw(image)
            for xmin, ymin, xmax, ymax, label in coords:
                xmin *= width
                xmax *= width
                ymin *= height
                ymax *= height
                draw.rectangle(xy=[xmin, ymin, xmax, ymax], outline="red")
                annot_text = '{}-{}'.format(str(label), text)
                draw.text(((xmin+xmax)//2-20, ymin+10), (annot_text), fill=(255, 0, 0))

        save_path = save_dir.joinpath(name_last)
        image.save(save_path)
        
def thread_fn_process_one_image(args):
    try:
        image_path, boxes, texts, labels, queue = args  
        tf_exmaple = process_one_image(image_path, boxes, texts, labels)
        queue.put(tf_exmaple)
    except Exception as e:
        print('error on image:', image_path.name)
        print(e)
        print()

def write_tfrecord(df: pd.DataFrame, images_dir: Path, output_dir: Path, num_workers: int, split: str, index: int):
    text_to_label = {
        'BI-RADS 3': 1,
        'BI-RADS 4': 2,
        'BI-RADS 5': 3,
    }
    output_filepath = output_dir.joinpath('vindr-mammo-{}-{}.tfrecord'.format(split, str(index)))
    tf_examples_queue = Queue()
    
    thread_args = []
    for index, row in df.iterrows():
        try: 
            image_path = images_dir.joinpath(row['image_id']+'.png')
            boxes_xmin = ast.literal_eval(row['boxes_xmin'].replace('nan', ''))
            boxes_ymin = ast.literal_eval(row['boxes_ymin'].replace('nan', ''))
            boxes_xmax = ast.literal_eval(row['boxes_xmax'].replace('nan', ''))
            boxes_ymax = ast.literal_eval(row['boxes_ymax'].replace('nan', ''))
            boxes = (boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax)
            texts = ast.literal_eval(row['finding_birads'].replace('nan', ''))
            labels = [1 for text in texts] #One class classification , for multiclass: [text_to_label[text] for text in texts]
            thread_args.append((image_path, boxes, texts, labels, tf_examples_queue))
        except Exception as e:
            error_message = traceback.format_exc()
            exception_info = "Exception Info:\n"+ str(error_message) # + '\n' + row['image_id'] + '\t' + row['boxes_xmin']  + '\t' + row['boxes_ymin'] + '\t' + row['boxes_xmax'] + '\t' + row['boxes_ymax'] + '\t' + row['finding_birads']
            print(exception_info)
            raise e
    """
    for arg in thread_args:
        thread_fn_process_one_image(arg)    
    """
    with ThreadPoolExecutor(num_workers) as executor:
        futures = executor.map(thread_fn_process_one_image, thread_args)
        for future in futures:
            try:
                # Access the result of the future (this will raise an exception if one occurred)
                result = future.result()
            except AttributeError as e:
                #print("Exception occurred:", e)
                pass
            except Exception as e:
                # Handle the exception here
                print("Exception occurred:", e)
                # You can also raise the exception again if you want to propagate it further
                raise e

    with tf.io.TFRecordWriter(str(output_filepath)) as writer:
        while not tf_examples_queue.empty():
            writer.write(tf_examples_queue.get().SerializeToString())

    return output_filepath
    
def split_df_into_pieces(df, num_train_pieces:int, num_validation_pieces:int):
    def random_split(df, N):
        df_shuffled = df.sample(frac=1, random_state=42)

        total_rows = df_shuffled.shape[0]
        rows_per_piece = total_rows // N

        pieces = [df_shuffled.iloc[i*rows_per_piece:(i+1)*rows_per_piece] for i in range(N-1)]
        pieces.append(df_shuffled.iloc[(N-1)*rows_per_piece:])
        return pieces
    
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    train_pieces = random_split(shuffled_df[shuffled_df['split'] == 'training'], num_train_pieces)
    validation_pieces = random_split(shuffled_df[shuffled_df['split'] == 'test'], num_validation_pieces)
    
    return (train_pieces, validation_pieces)

def thread_fn_write_tfrecord(args):
    df, images_dir, output_dir, threads_per_piece_writer, split, index, queue = args
    record_path = write_tfrecord(df, images_dir, output_dir, threads_per_piece_writer, split, index)
    queue.put(record_path)

def main(images_dir: Path, ouput_dir: Path, annotations_csv_filepath: Path, visualize: int, num_train_pieces:int, num_validation_pieces:int, num_workers: int):

    train_records_queue = Queue()    
    validation_records_queue = Queue()    
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(annotations_csv_filepath)

    train_pieces, validation_pieces = split_df_into_pieces(df, num_train_pieces, num_validation_pieces)
    
    thread_args = []
    num_total_files = num_train_pieces + num_validation_pieces
    threads_per_piece_writer = (num_workers - num_total_files) // num_total_files
    
    for index, p in enumerate(train_pieces):
        thread_args.append((p, images_dir, ouput_dir, threads_per_piece_writer, 'train', index, train_records_queue))
    
    for index, p in enumerate(validation_pieces):
        thread_args.append((p, images_dir, ouput_dir, threads_per_piece_writer, 'validation', index, validation_records_queue))
    
    """
    for arg in thread_args:
        thread_fn_write_tfrecord(arg)
    """
    
    # Create a thread pool with a maximum of num_workers threads
    with ThreadPoolExecutor(max_workers=num_total_files) as executor:
        # Submit the tasks to the thread pool
        futures = executor.map(thread_fn_write_tfrecord, thread_args)
        
        # Iterate over the futures to check for exceptions
        for future in futures:
            try:
                # Access the result of the future (this will raise an exception if one occurred)
                result = future.result()
            except AttributeError as e:
                #print("Exception occurred:", e)
                pass
            except Exception as e:
                # Handle the exception here
                print("Exception occurred:", e)
                # You can also raise the exception again if you want to propagate it further
                raise e
    while not train_records_queue.empty():
        record_path = train_records_queue.get()
        save_dir = ouput_dir.joinpath('visualizations', 'train', record_path.stem)
        os.makedirs(save_dir)
        visualize_tfrecords(record_path, save_dir, visualize)

    while not validation_records_queue.empty():
        try:
            record_path = validation_records_queue.get()
            save_dir = ouput_dir.joinpath('visualizations', 'validation', record_path.stem)
            os.makedirs(save_dir)
            visualize_tfrecords(record_path, save_dir, visualize)
        except Exception as e:
            print(e)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts xml annotations and images to tfrecord file')
    parser.add_argument('-a', '--annotations-csv', required=True, type=str,
                        help='Path to csv file which contains annotations')
    parser.add_argument('-i', '--images-dir', required=True, type=str,
                        help='Folder path of image files', default='/data/emre/ms/vindr/dataset/cropped/')
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='Path to outputs to be saved. Needs to be folder, not file path', default='/data/emre/ms/vindr/dataset/tfrecords/')
    parser.add_argument('-v', '--visualize', required=False, type=int, default=10,
                        help='Number of images to be visualized. default:10')
    parser.add_argument('-p', '--tfrecord-pieces', required=False, type=str, default='3,1',
                        help='How many pieces that tfrecords splits into, should bes passed as train_pieces,valid_pieces. Example: "2,3"')
    parser.add_argument('-n', '--num-workers', required=False, type=int, default=120,
                        help='Number of workers can run parallely')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    annotations_csv_filepath = Path(args.annotations_csv)
    num_train_pieces, num_validation_pieces = list(map(int, args.tfrecord_pieces.split(',')))
    if not images_dir.is_file:
        raise NotADirectoryError(images_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    if not annotations_csv_filepath.is_file:
        raise FileNotFoundError(annotations_csv_filepath)

    main(images_dir, output_dir, annotations_csv_filepath, args.visualize, num_train_pieces, num_validation_pieces, args.num_workers)
    # TODO: check bbox error on  overlapping annotations, check 3_test visualizations