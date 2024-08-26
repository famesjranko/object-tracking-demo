"""
Script for real-time object detection across multiple RTSP video streams using 
YOLO models, with multiprocessing. It maps locations to their stream URLs, 
defines YOLO model weights for varying accuracy and computational needs, and 
sets detection thresholds for different object classes. A function constructs 
command line arguments for object detection, and another processes each stream 
with specified model weights and detection settings. Uses multiprocessing to 
handle each video stream in parallel, optimizing for real-time performance in 
surveillance or monitoring tasks.
"""

import subprocess
import multiprocessing

# Define the streams (rooms) from meraki cameras
streams = {
    'OpenDeskSpace203': 'rtsp://192.168.3.87:9000/live',
    'CollaborationHub201C': 'rtsp://192.168.3.75:9000/live',
    'MV12-EntryDoor': 'rtsp://192.168.3.85:9000/live',
    'PresentationEntry200': 'rtsp://192.168.3.76:9000/live',
    'PitchSpace213': 'rtsp://192.168.3.69:9000/live',
}

# Define YOLO model weights
weights = {
    'nano': 'yolov5n.pt',
    'small': 'yolov5s.pt',
    'medium': 'yolov5m.pt',
    'large': 'yolov5l.pt',
    'extra_large': 'yolov5x.pt',
}

# Define wanted Class detection IDs and their confidence levels
class_confidences = {
     0: 0.55,   # person
    13: 0.55,   # bench
    24: 0.55,   # backpack
    26: 0.55,   # handbag
    56: 0.55,   # chair
    57: 0.55,   # couch
    60: 0.55,   # dining table
    62: 0.55,   # tv
    63: 0.55,   # laptop
    65: 0.55,   # keyboard
}

# Build command arguments for classes and their confidences
def build_class_conf_args(classes_confidences):
    classes = [str(cls) for cls in classes_confidences.keys()]
    confidences = [str(conf) for conf in classes_confidences.values()]
    return ['--classes'] + classes + ['--conf-thres-per-class'] + confidences

# Set the baseline confidence level
general_conf = ['--conf-thres', '0.25']

# Function to process a camera stream
def process_stream(room, url):
    cmd = [
        'python', 'obj_det_and_trk-many.py',
        '--weights', weights['small'],
        '--source', url,
        '--video_name', room,
    ] + general_conf + build_class_conf_args(class_confidences)
    
    print(f"calling: {cmd}\n")

    subprocess.call(cmd)

if __name__ == "__main__":
    # Create a multiprocessing pool with the number of processes you want
    pool = multiprocessing.Pool(processes=len(streams))

    # Use the pool to execute the process_stream function for each camera stream
    results = [pool.apply_async(process_stream, (room, url)) for room, url in streams.items()]

    # Wait for all processes to complete
    pool.close()
    pool.join()
