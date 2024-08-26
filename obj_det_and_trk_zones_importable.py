"""
Author: Andrew J McDonald
Date: 17.04.2023
---------------Things to do-----------------
  1. save zones to text                                                     [done!]
  2. load zones from text                                                   [done!]
  3. find a way to keep mouse rectangle visible through loop                [done!]
  4. account for overlap zone areas as unique?                              [     ]
  5. track only in roi?                                                     [done!] set by bool: track_in_zones_only = True
  6. do some counting and storing of objects over time, instances? SQLite?  [     ]
  7. track tables chairs? - find office type of crowd video                 [     ]
  8. differentiate people sitting down vs standing up?                      [     ]
  9. heatmap?                                                               [     ]
 10. multiple images, crop together, then feed to detector                  [     ] next!
 
 Updates:
  a. ADDED testing for per class confidence thresholds, 
    see '# --------- TESTING 1 --------- # sections'
    
    example to use:
    python obj_det_and_trk_zones.py --weights yolov5s.pt --source 0 --classes 0 56 --conf-thres-per-class 0.7 0.6 --conf-thres 0.25
    
    this runs person with confidence of 70% and chair with confidence of 70%; with detector predictor of 25%
    Meaning, that detections will be found that are above 25% from the detector, but the script will only be interested in recording them
    when a person is over 70% and chair is over 60%.  
    
      So --conf-thres is the lowest threshold for the detector and --conf-thres-per-class is a relative threshold for each class labeled.
    
  b. ADDED visualisation improvements - seperate zone and track visual function and moved visualise_objects function into tracks loop
  c. ADDED recording of 1m, 1hr, 12hr, and 24hr unique id counts to each zone
----------------Example---------------------
python obj_det_and_trk_zones.py --weights yolov5n.pt --source "videos/crowd-1.mp4" --classes 0 --conf-thres .25 --load-zones zone_save_test.txt --save-zones zone_save_test.txt
"""

import os
import time
import datetime

import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import string
import random


#---------------Global Variables---------------
drawing = False
x, y, w, h = 0, 0, 0, 0

# Define an empty list to hold the dynamic zones
zone_rois = []
zone_counter = 0

save_zones_path = None
load_zones_path = None


#---------------Object Tracking---------------
import skimage
from sort import *


#-----------Object Blurring-------------------
blurratio = 40


#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


#.................. Zone Functions .................

# This function checks whether the center point of a box is inside a given ROI
# (region of interest), which is defined by a rectangle. It returns True if the
# center point is inside the ROI, and False otherwise.
def is_inside(box, roi):
    if len(box) == 4:
        xmin, ymin, xmax, ymax = box
        cx, cy = int((xmin+xmax)/2), int((ymin+ymax)/2)  # calculate center point of box
    elif len(box) == 2:
        cx, cy = int(box[0]), int(box[1]) # is the center point of box
        
    rx, ry, rw, rh = roi
    return rx <= cx <= rx+rw and ry <= cy <= ry+rh
    
# Creates a dictionary representing a detection zone with the given ID, color, and ROI.
# The dictionary contains information about the zone, including unique ID, data about
# objects detected over time, and data for collecting detections counts at regular intervals.
def create_zone(zone_id, zone_color, zone_roi):
    return {
        'id': zone_id,
        'color': zone_color,
        'roi': zone_roi,
        'zone_dict': {
            'object_count': 0,
            'unique_ids': set(),
        },
        'types': [],
        'data': {
            'num_objects_over_time': [], # list for recorded interval data
            'unique_ids_1m': set(),
            'unique_ids_1hr': set(),
            'unique_ids_12hr': set(),
            'unique_ids_24hr': set(),
        },
        'data_intervals': { # keeps track of start time for current interval
            'last_1m': datetime.datetime.fromtimestamp(time.time()),
            'last_1hr': datetime.datetime.fromtimestamp(time.time()),
            'last_12hr': datetime.datetime.fromtimestamp(time.time()),
            'last_24hr': datetime.datetime.fromtimestamp(time.time()),
        },
    }

# This is a function that creates a rectangle using mouse events in OpenCV.
# It captures left and right mouse button events to draw and remove rectangles.
# It also generates unique ids for the drawn zones, randomly assigns colors to#
# them, and saves them to a file if specified.
def draw_rectangle(event, x_new, y_new, flags, params):
    global x, y, w, h, drawing, zone_rois, zone_counter, save_zones
    
    # Capture the mouse down event and initialize the rectangle drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = x_new, y_new

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x_new - x, y_new - y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x_new - x, y_new - y

        # Swap the coordinates if the user started dragging from the bottom right corner
        if w < 0:
            x, x_new = x_new, x
            w *= -1
        if h < 0:
            y, y_new = y_new, y
            h *= -1

        if w >= 20 and h >= 20:
            # Generate a unique id for the zone
            zone_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

            # Check if the id is already used, and generate a new one if necessary
            while zone_id in [zone['id'] for zone in zone_rois]:
                zone_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

            # generate random RGB values
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)

            # create a tuple of random RGB values
            zone_color = (r, g, b)

            # create the zone dictionary with the random color
            zone = create_zone(zone_id, zone_color, (x, y, w, h))
            
            # append zone to zones
            zone_rois.append(zone)
            print(f'Added {zone["id"]}: {zone["roi"]}')
            
            # Reset the rectangle coordinates
            x, y, x_new, y_new, w, h = 0, 0, 0, 0, 0, 0
    
    # Capture the right mouse button down event and remove the zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Calculate the distance between the center of each zone and the current mouse position
        distances = [(zone['roi'][0] + zone['roi'][2] / 2 - x_new) ** 2 + (zone['roi'][1] + zone['roi'][3] / 2 - y_new) ** 2 for zone in zone_rois]

        if len(distances) > 0:
            # Find the index of the zone with the smallest distance
            closest_index = distances.index(min(distances))
        
            # Remove the zone with the smallest distance
            print(f'Removed {zone_rois[closest_index]["id"]}: {zone_rois[closest_index]["roi"]}')
            del zone_rois[closest_index]
            
    # Save zones to file if specified
    if save_zones_path is not None and (event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONUP):
        # Open the file for writing
        with open(save_zones_path, 'w') as f:
            # Loop over each zone in the list of zones
            for zone in zone_rois:
                # qWrite the zone's id, color, and roi to the file
                # using string formatting with curly braces
                f.write(f"{zone['id']} {zone['color']} {zone['roi']}\n")
                print(f"Saving zone: id:{zone['id']} color:{zone['color']} roi:{zone['roi']}")
        # Print a message indicating the number of zones that were saved
        # and the path to the file where they were saved
        if len(zone_rois) >= 1:
            print(f"Saved {len(zone_rois)} zones to {save_zones_path}")
        else:
            print(f"No zones saved to {save_zones_path}")

#.................. Visualisation Functions .................

# This function takes an image and draws lines for tracked objects
def draw_tracks(im0, track, color, zone_roi=None):
    [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
        (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
        color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
        if i < len(track.centroidarr)-1 and 
        (zone_roi is None or is_inside(track.centroidarr[i], zone_roi['roi']) or is_inside(track.centroidarr[i+1], zone_roi['roi']))]


# This function takes an image and a list of zone rois and draws rectangles around the zones on the image.
# It also displays text information about each zone, including its ID, the number of objects in the zone, and the number of unique IDs.
# The function uses OpenCV's rectangle and putText functions to draw the rectangles and text.
def draw_zones(im0, zone_rois):
    for zone_roi in zone_rois:
        zone_dict = zone_roi['zone_dict']

        # draw zone roi and display zone info
        cv2.rectangle(im0, (zone_roi['roi'][0], zone_roi['roi'][1]), 
                      (zone_roi['roi'][0]+zone_roi['roi'][2],zone_roi['roi'][1]+zone_roi['roi'][3]), 
                      zone_roi['color'], 2)

        # first line of text
        text_line1 = f"{zone_roi['id']}, {zone_dict['object_count']} objects, {len(zone_dict['unique_ids'])} unique IDs"
        cv2.putText(im0, text_line1, (zone_roi['roi'][0], zone_roi['roi'][1]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_roi['color'], 2)

        # second line of text
        text_line2 = f"{zone_roi['types']}"
        cv2.putText(im0, text_line2, (zone_roi['roi'][0], zone_roi['roi'][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_roi['color'], 2)

        
# This function visualizes the tracking results of objects in different zones of an image. 
# It takes in the tracked detections, zone ROIs, a category dictionary, and the original image. 
# For each tracked detection, it checks if it is inside a zone ROI and updates the zone
# information accordingly. Then it draws bounding boxes, labels, and information for each zone
# ROI and displays the resulting image.
def visualize_tracking_results(im0, tracked_dets, zone_rois, category_dict, track_in_zones_only, track=None):
    bbox_xyxy  = tracked_dets[:,:4]  # extract the bounding box coordinates in XYXY format
    identities = tracked_dets[:, 8] # extract the identities of the objects
    categories = tracked_dets[:, 4] # extract the categories of the objects
    confidence = tracked_dets[:, 5] # extract the confidence scores of the detections
    #offset=(0, 0)                  # set the offset for the bounding boxes (here, there is no offset)
    
    if not track_in_zones_only:
        # draw tracked object lines on frame
        draw_tracks(im0, track, (0,255,0))
    
    # loop through all zone rois
    for zone_roi in zone_rois:
        zone_dict = zone_roi['zone_dict']
        #print(track)
        
        # Reset the object count for zone
        zone_dict['object_count'] = 0

        # loop through all tracked detections
        for bbox, identity, category in zip(bbox_xyxy, identities, categories):
            # check if detection is inside zone roi
            if is_inside(bbox, zone_roi['roi']):
                zone_dict['object_count'] += 1
                
                zone_dict['unique_ids'].add(identity)
                zone_roi['data']['unique_ids_1m'].add(identity)
                zone_roi['data']['unique_ids_1hr'].add(identity)
                zone_roi['data']['unique_ids_12hr'].add(identity)
                zone_roi['data']['unique_ids_24hr'].add(identity)
                
                #print(zone_roi['data']['unique_ids_15s'])
                #print(zone_roi['data']['unique_ids_1hr'])
                #print(zone_roi['data']['unique_ids_12hr'])
                #print(zone_roi['data']['unique_ids_24hr'])
                
                # Extract the bounding box coordinates and convert them to integers
                x1, y1, x2, y2 = [int(i) for i in bbox]
                
                # Shift the box by n pixels in the x1,x2,y1,y2 axis
                x1 += 0
                x2 += 0
                y1 += 0
                y2 += 0
                
                # Convert category to an integer if it's not None, else set it to 0
                cat = int(category) if category is not None else 0
                
                # Get the name of the category from the category dictionary
                category_name = category_dict[cat]
                
                # Convert identity to an integer if it's not None, else set it to 0
                id = int(identity) if identity is not None else 0
                
                # Print object id and category to terminal
                #print('identity, category:', identity, category_name)
                
                # Get the center point of the bounding box
                center_point = (int((bbox[0]+bbox[2])/2),(int((bbox[1]+bbox[3])/2)))
                
                # Create the label for the object
                object_label = f"{id} [{category_name}]" if category is not None else str(id)
                
                # Get the size of the label text
                (w, h), _ = cv2.getTextSize(object_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Draw the bounding box, background, label, and a circle at the center of the bounding box
                cv2.rectangle(im0, (x1, y1), (x2, y2), zone_roi['color'], 2)
                cv2.rectangle(im0, (x1, y1 - 20), (x1 + w, y1), zone_roi['color'], -1)
                cv2.putText(im0, object_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
                cv2.circle(im0, center_point, 3, zone_roi['color'],-1)
                
                # If the category name is not in the list, append it to the zone_roi's types list
                if category_name not in zone_roi['types']:
                    #print(category_name)
                    zone_roi['types'].append(category_name)
                
                # draw tracked object lines on frame
                draw_tracks(im0, track, zone_roi['color'], zone_roi)


#..............................................................................


#@torch.no_grad()
# def detect(weights=ROOT / 'yolov5n.pt',
        # source=ROOT / 'yolov5/data/images', 
        # data=ROOT / 'yolov5/data/coco128.yaml',  
        # imgsz=(640, 640), conf_thres=0.25, conf_thres_per_class=None,
        # iou_thres=0.45, max_det=1000, device='cpu',  view_img=True,  
        # save_txt=False, save_conf=False, save_crop=False, 
        # nosave=False, classes=None,  agnostic_nms=False,  
        # augment=False, visualize=False,  update=False,  
        # project=ROOT / 'runs/detect',  name='exp',  
        # exist_ok=False, line_thickness=2,hide_labels=False,  
        # hide_conf=False,half=False,dnn=False,display_labels=False,
        # blur_obj=False,color_box = False,
        # save_zones=None, load_zones=None):
    
    # global x, y, w, h, drawing, zone_rois, zone_counter, save_zones_path, load_zones_path
    
    # source = str(opt.source)  # convert to string
    # save_img = not nosave and not source.endswith('.txt')
    
    # print(source)

# edited the function parameters to work more easily with module importing
# can set defaults here, or set on function call
@torch.no_grad()
def detect(weights=ROOT / 'yolov5s.pt',
           source=ROOT / 'data/images',
           data=ROOT / 'data/coco128.yaml',
           imgsz=(640, 640),
           conf_thres=0.25,
           conf_thres_per_class=None,
           iou_thres=0.45,
           max_det=1000,
           device='cpu',
           view_img=True,
           save_txt=False,
           save_conf=False,
           save_crop=False,
           nosave=False,
           classes=None,
           agnostic_nms=False,
           augment=False,
           visualize=False,
           update=False,
           project=ROOT / 'runs/detect',
           name='exp',
           exist_ok=False,
           line_thickness=2,
           hide_labels=False,
           hide_conf=False,
           half=False,
           dnn=False,
           blur_obj=False,
           color_box=False,
           save_zones=None,
           load_zones=None):
    
    global x, y, w, h, drawing, zone_rois, zone_counter, save_zones_path, load_zones_path
    
    source = str(source)  # convert to string
    save_img = not nosave and not source.endswith('.txt')
    
    if save_zones is not None:
        save_zones_path = str(Path(opt.save_zones))
        
    if load_zones is not None: 
        load_zones_path = str(Path(opt.load_zones))
    
    # --------- TESTING 1 --------- #
    use_conf_thres_per_class = False
    print('conf_thres',conf_thres)
    print('conf_thres_per_class',conf_thres_per_class)
    
    # set per class confidence threshold array and flag
    if conf_thres_per_class is not None:
        if classes is not None:
            use_conf_thres_per_class = True
            conf_thres_per_class = {c: conf_thres_per_class[i] for i, c in enumerate(classes)}

            if len(classes) != len(conf_thres_per_class):
                print("Error: The size of --classes and --conf-thres-2 lists must be the same.")
                exit()
            print('conf_thres_per_class:', conf_thres_per_class)
        else:
            print("NOTICE: because classes was not set via --classes, --conf-thres-per-class is being ignored.")
    
    print('use_conf_thres_per_class:',use_conf_thres_per_class)
    # --------- TESTING 1 --------- #
    
    #.... Initialize SORT .... 
    # sort_max_age controls the maximum number of frames that a tracked object can be 
    # "lost" (not detected) before the tracker stops tracking it. A higher value allows 
    # for longer-term tracking, but increases the risk of tracking irrelevant objects.
    sort_max_age = 30 

    # sort_min_hits controls the minimum number of consecutive detections required to 
    # initiate a track for a new object. A higher value reduces the risk of false 
    # detections leading to new tracks, but also increases the risk of not detecting 
    # objects that appear briefly.
    sort_min_hits = 2

    # sort_iou_thresh controls the minimum IOU (Intersection over Union) threshold 
    # required for two detections to be considered as part of the same object. A higher 
    # value reduces the risk of merging detections from different objects, but also 
    # increases the risk of not merging detections from the same object that are 
    # slightly misaligned or partially occluded.
    sort_iou_thresh = 0.2

    # sort_tracker is the instance of the Sort tracker, initialized with the 
    # parameters above.
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    #......................... 
    
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
        
    image = source.lower().endswith('.jpg') or source.lower().endswith('.jpeg') or source.lower().endswith('.png')
        
    #print('webcam', webcam)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
    if pt or jit:
        model.model.half() if half else model.model.float()

    if webcam:
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset) 
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1 
    
    vid_path, vid_writer = [None] * bs, [None] * bs
    t0 = time.time()
    dt, seen = [0.0, 0.0, 0.0], 0
    
    # Load coco object labels from text file
    category_dict = {}
    with open('coco_labels.txt', 'r') as f:
        for line in f:
            index, label = line.strip().split(maxsplit=1)
            category_dict[int(index)] = label
   
    # Load zones from file if specified
    if load_zones_path is not None:   # Check if a path to a zones file was specified
        with open(load_zones_path, 'r') as f:   # Open the file at the specified path
            lines = f.readlines()   # Read in all the lines in the file
            for line in lines:   # Loop through each line in the file
                # Parse the zone ID, color, and ROI from the line
                zone_id, zone_color_roi = line.strip().split(' ', maxsplit=1)
                zone_color_str, zone_roi_str = zone_color_roi.strip('()').split(') (')
                
                # Convert the color and ROI strings to tuples of integers
                zone_color = tuple(int(x) for x in zone_color_str.split(','))
                zone_roi = tuple(int(x) for x in zone_roi_str.split(','))
                
                # Create a new zone dictionary with the parsed values and append it to the list of zones
                zone = create_zone(zone_id, zone_color, zone_roi)
                zone_rois.append(zone)

    # set default and exit flags
    default_zone_set = False
    exit_flag = False

    # show video feed frame
    show_frame = True
    
    # tracking in zones only
    track_in_zones_only = True
    
    # set the interval to record data (in seconds)
    data_interval = 60

    # initialize time variables
    last_recorded_time = datetime.datetime.fromtimestamp(time.time())
    
    # Create a window with the same name as the video feed
    cv2.namedWindow('frame')
    
    # Set up the mouse callback function -TESTING
    cv2.setMouseCallback('frame', draw_rectangle)
    
    print("Inferencing, starting...")
    for path, im, im0s, vid_cap, s in dataset:
        # get the current time
        current_time = datetime.datetime.fromtimestamp(time.time())
        
        # =====================================================
        # original version of the interval data list 
        # =====================================================

        #if len(zone_rois) > 0:
            ## Record data for each zone
            #for zone in zone_rois:
                #zone_data = zone['data']
                #zone_inteval_times = zone['data_intervals']

                ## Record instantaneous number of objects
                #timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                ## reset interval counts
                #interval_1m = 0
                #interval_1hr = 0
                #interval_12hr = 0
                #interval_24hr = 0

                ## Collected data for each interval period and record to zone
                #if current_time - zone_inteval_times['last_1m'] >= datetime.timedelta(seconds=60):
                    #interval_1m = len(zone_data['unique_ids_1m'])
                    #zone_data['unique_ids_1m'] = set()
                    ##last_1m_time = current_time
                    #zone_inteval_times['last_1m'] = current_time

                    #if current_time - zone_inteval_times['last_1hr'] >= datetime.timedelta(hours=1):
                        #interval_1hr = len(zone_data['unique_ids_1hr'])
                        #zone_data['unique_ids_1hr'] = set()
                        ##last_1hr_time = current_time
                        #zone_inteval_times['last_1hr'] = current_time

                        #if current_time - zone_inteval_times['last_12hr'] >= datetime.timedelta(hours=12):
                            #interval_12hr = len(zone_data['unique_ids_12hr'])
                            #zone_data['unique_ids_12hr'] = set()
                            ##last_12hr_time = current_time
                            #zone_inteval_times['last_12hr'] = current_time
                        
                            #if current_time - zone_inteval_times['last_24hr'] >= datetime.timedelta(hours=24):
                                #interval_24hr = len(zone_data['unique_ids_24hr'])
                                #zone_data['unique_ids_24hr'] = set()
                                ##last_24hr_time = current_time
                                #zone_inteval_times['last_24hr'] = current_time
                    
                    ## Record data for each interval to zone
                    #zone_data['num_objects_over_time'].append((timestamp, interval_1m, interval_1hr, interval_12hr, interval_24hr))
                    #print(zone_data['num_objects_over_time'])
            
        # =====================================================
        # testing a different version of the interval data list 
        # =====================================================
        if len(zone_rois) > 0:
            for zone in zone_rois:
                zone_data = zone['data']
                zone_interval_times = zone['data_intervals']

                # Record instantaneous number of objects
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Reset interval counts
                interval_1m = None
                interval_1hr = None
                interval_12hr = None
                interval_24hr = None

                # Collect data for each interval period and record to zone
                if current_time - zone_interval_times['last_1m'] >= datetime.timedelta(seconds=60):
                    interval_1m = len(zone_data['unique_ids_1m'])
                    zone_data['unique_ids_1m'] = set()
                    zone_interval_times['last_1m'] = current_time

                    if current_time - zone_interval_times['last_1hr'] >= datetime.timedelta(hours=1):
                        interval_1hr = len(zone_data['unique_ids_1hr'])
                        zone_data['unique_ids_1hr'] = set()
                        zone_interval_times['last_1hr'] = current_time

                        if current_time - zone_interval_times['last_12hr'] >= datetime.timedelta(hours=12):
                            interval_12hr = len(zone_data['unique_ids_12hr'])
                            zone_data['unique_ids_12hr'] = set()
                            zone_interval_times['last_12hr'] = current_time

                            if current_time - zone_interval_times['last_24hr'] >= datetime.timedelta(hours=24):
                                interval_24hr = len(zone_data['unique_ids_24hr'])
                                zone_data['unique_ids_24hr'] = set()
                                zone_interval_times['last_24hr'] = current_time

                    # Record data for each interval to zone
                    zone_data['num_objects_over_time'].append({'timestamp': timestamp, 'interval_data': {'interval_1m': interval_1m, 'interval_1hr': interval_1hr, 'interval_12hr': interval_12hr, 'interval_24hr': interval_24hr}})
                    print(zone_data['num_objects_over_time'])
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
            
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # reset zone object count
        for zone_roi in zone_rois:
            zone_dict = zone_roi['zone_dict']
            zone_dict['object_count'] = 0
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
            # set the maximum size for width or height of the visualised frame
            max_size = 1000
            
            # calculate the scaling factor to maintain aspect ratio
            scale_factor = min(1, max_size / max(im0.shape[:2]))
            
            # resize the frame
            im0 = cv2.resize(im0, (int(im0.shape[1] * scale_factor), int(im0.shape[0] * scale_factor)))
            
            # add 'default' zone of entire frame if no zones loaded
            if (load_zones_path is None or len(zone_rois) < 1) and not default_zone_set:
                print('creating default zone')
                
                frame_height, frame_width = im0.shape[:2]
                zone = create_zone('default', (0, 255, 0), (0, 0, frame_width, frame_height))
                print('zone:', zone)
                
                zone_rois.append(zone)
                print(f"zone {zone['id']} added to zones list")
                
                default_zone_set = True
                
            # --------- TESTING 3 --------- #
            # this is working!
            # Draw the rectangle on the video feed
            if drawing:
                cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            # Create the dynamic zones based on the rectangle drawn by the user
            if not drawing and w != 0 and h != 0:
                x, y, w, h = 0, 0, 0, 0
            # --------- TESTING 3 --------- #
                
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if blur_obj:
                        crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                        blur = cv2.blur(crop_obj,(blurratio,blurratio))
                        im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    else:
                        continue
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    #dets_to_sort = np.vstack((dets_to_sort, 
                    #                          np.array([x1, y1, x2, y2, 
                    #                                    conf, detclass])))
                
                # --------- TESTING 1 --------- #
                    #print('detclass:', int(detclass))
                    if not use_conf_thres_per_class:
                        #print('in use_conf_thres_per_class as FALSE!')
                        dets_to_sort = np.vstack((dets_to_sort, 
                                                  np.array([x1, y1, x2, y2, conf, detclass])))
                    else:
                        # Check if the detection meets the confidence threshold for the class
                        if int(detclass) not in conf_thres_per_class:
                            #print('Class not found in conf_thres_per_class:', int(detclass), conf_thres_per_class)
                            continue
                        elif conf >= conf_thres_per_class[int(detclass)]:
                            #print('Detection meets confidence threshold:', category_dict[int(detclass)], conf, conf_thres_per_class[int(detclass)])
                            dets_to_sort = np.vstack((dets_to_sort, 
                                                      np.array([x1, y1, x2, y2, conf, detclass])))
                        #else:
                            #print('Detection does not meet confidence threshold:', category_dict[int(detclass)], conf, conf_thres_per_class[int(detclass)])
                                                        
                # --------- TESTING 1 --------- #
                                                        
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()     

                # loop over tracks
                for track in tracks:
                    if len(tracked_dets)>0:
                        # draw objects boundary box on frame
                        visualize_tracking_results(im0,
                                                   tracked_dets,
                                                   zone_rois,
                                                   category_dict,
                                                   track_in_zones_only,
                                                   track)
        
        # draw zone boundary box on frame
        draw_zones(im0, zone_rois)
        
        
                
        if image:
            # if dataset.mode == 'image':
            #cv2.imwrite(f"output/{source}", im0)
            return im0
            # else:
                # if vid_path != save_path: 
                    # vid_path = save_path
                    # if isinstance(vid_writer, cv2.VideoWriter):
                        # vid_writer.release()  
                    # if vid_cap: 
                        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # else:
                        # fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # save_path += '.mp4'
                    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                # vid_writer.write(im0)
        elif show_frame:
            cv2.imshow('frame', im0)
            
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                exit_flag = True
                break
                    
        if exit_flag:
            break
            
    print("Inferencing, stopped.")
    cv2.destroyAllWindows()

    if update:
        strip_optimizer(weights)
    
    if vid_cap:
        vid_cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.20, help='confidence threshold')
    parser.add_argument('--conf-thres-per-class', nargs='+', type=float, default=None,
                    help='list of confidence thresholds, one for each class (default: [0.2]). Example usage: --conf-thres-2 0.5 0.6 0.3')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--blur-obj', action='store_true', help='Blur Detected Objects')
    parser.add_argument('--color-box', action='store_true', help='Change color of every box and track')
    parser.add_argument('--load-zones', type=str, default=None, help='text file containing zones to load')
    parser.add_argument('--save-zones', type=str, default=None, help='text file to save dynamically created zones')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    
