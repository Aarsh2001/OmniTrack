import os
import sys
os.chdir('../AiATrack')
sys.path.append(os.path.abspath('../AiATrack'))


import json
import torch
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from collections import OrderedDict
import importlib
import cv2 as cv
import glob
import time
from vot.lib.omni import *
from vot.lib.utils import *
import logging

# Initialize the logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('process_log.txt')]
)

logger = logging.getLogger(__name__)


# Load the parameter module and tracker class
param_module = importlib.import_module('lib.test.parameter.aiatrack')
params = param_module.parameters('baseline')
tracker_module = importlib.import_module('lib.test.tracker.aiatrack')
tracker_class = tracker_module.get_tracker_class()

# Path to your sequences
sequences_path = '/mnt/data_f_500/aarsh/data/Test_3840x1920/'
output_path = '../default_revamped_benchmark_4/'
fps_path = '../fps/'
log_file = os.path.join(output_path, 'error_log.txt')

# Get available GPUs
available_gpus = list(range(torch.cuda.device_count()))

def _read_image(image_file: str):
    if isinstance(image_file, str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    else:
        raise ValueError('ERROR: type of image_file should be str')

def json2bbox(json_data):
    """Extract bbox from the JSON data."""
    bbox = dict2Bbox(json_data)
    return list(bbox.tolist_xywh())
    

def get_search_crop(frames, init_info, tracker, seq_name=None, threshold=20, FOV=90, kernel_size=500, apply_method=False):
    output = {'target_bbox': [], 'time': []}

    if tracker.params.save_all_boxes:
        output['all_boxes'] = []
        output['all_scores'] = []

    def _store_outputs(tracker_out: dict, defaults=None):
        defaults = dict() if defaults is None else defaults
        for key in output.keys():
            val = tracker_out.get(key, defaults.get(key, None))
            if key in tracker_out or val is not None:
                output[key].append(val)

    # Initialize
    image = _read_image(frames[0])
    start_time = time.time()
    out = tracker.initialize(image, init_info, seq_name=seq_name)
    if out is None:
        out = dict()

    init_default = {'target_bbox': init_info.get('init_bbox'), 'time': time.time() - start_time}
    if tracker.params.save_all_boxes:
        init_default['all_boxes'] = out['all_boxes']
        init_default['all_scores'] = out['all_scores']

    _store_outputs(out, init_default)
    
    start_time = time.time()
    for frame_num, frame_path in enumerate((frames[1:]), start=1):
        image = _read_image(frame_path)
        out = tracker.track(image, seq_name=seq_name, threshold=threshold, FOV=FOV, apply_method=apply_method) 
        _store_outputs(out, {'time': time.time() - start_time})
    total_time = time.time() - start_time
    fps = len(frames) / total_time
    return output, fps


def process_sequence(seq_folder, gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
        tracker = tracker_class(params, 'default', debug=False)
        seq_num = os.path.basename(seq_folder)
        frames = sorted(glob.glob(os.path.join(seq_folder, 'image/*.jpg')))
        with open(os.path.join(seq_folder, 'label.json')) as f:
            json_data = json.load(f)
        
        init_info = {'init_bbox': json2bbox(json_data["000000.jpg"]['bbox'])}
#         logger.info(f"Initialization info: {init_info}")
        
        threshold_val = 20  # Set your desired threshold value
        FOV = 90  # Set your desired FOV value

        
        output_dict, fps = get_search_crop(frames, init_info, tracker, f'{seq_num}_lasotext', threshold=threshold_val, FOV=FOV, apply_method=False)
        logger.info(f"Finished Processing for {seq_folder} on GPU {gpu_id}")
        output_file = os.path.join(output_path, f'{seq_num}.txt')
        
        with open(output_file, 'w') as file:
            for i, sublist in enumerate(output_dict['target_bbox']):
                line = ' '.join(map(str, sublist))
                file.write(line)
                if i < len(output_dict['target_bbox']) - 1:
                    file.write('\n')

        # Output FPS information
        fps_file = os.path.join(fps_path, f'{seq_num}_fps.txt')
        with open(fps_file, 'w') as file:
            file.write(f"FPS: {fps:.2f}\n")

    except Exception as e:
        # Log the error and skip the sequence
        logger.error(f"Error processing sequence {seq_folder} on GPU {gpu_id}: {str(e)}")

def main():
    logger.info(f"Start benchmarking")
    seq_folders = sorted(glob.glob(os.path.join(sequences_path, '*')))
    # Split sequences among GPUs
    num_gpus = len(available_gpus)
    chunks = [seq_folders[i::num_gpus] for i in range(num_gpus)]    
    tasks = [(seq_folder, gpu_id) for gpu_id, seq_chunk in enumerate(chunks) for seq_folder in seq_chunk]

    with Pool(num_gpus) as pool:
        pool.starmap(process_sequence, tasks)
    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    main()