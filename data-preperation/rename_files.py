from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

def rename_file(filepath: Path):
    new_name = str(filepath.name).split('_')[1]
    new_path = filepath.parent.joinpath(new_name)
    filepath.rename(new_path)

def main(input_dir: Path, output_dir: Path, num_workers = None):
    filepaths = [path for path in input_dir.glob('*.png')]

    # Create a thread pool with a maximum of num_workers threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Prepare the arguments for each file to be processed
        # Submit the tasks to the thread pool
        executor.map(rename_file, filepaths)


if __name__ == '__main__':
    parser = ArgumentParser('Rename Files')
    parser.add_argument('-i', '--input-dir', required=True, type=str, help='')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='Output direcotory')
    parser.add_argument('-n', '--num-workers', required=False, type=int, default=60, help='Pass this parameter to overwrite existing files')

    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_file:
        raise NotADirectoryError(input_dir)
    if not output_dir.is_dir:
        raise NotADirectoryError(output_dir)
    
    main(input_dir, output_dir, args.num_workers)
