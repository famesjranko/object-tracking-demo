# yolov5-object-tracking

### New Features
- YOLOv5 Object Tracking Using Sort Tracker
- Added Object blurring Option
- Added Support of Streamlit Dashboard
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported
- Blacked demo (dots on black background
- Streamlit web application (app.py)

### Coming Soon
- Option to crop and save detected objects
- Dashboard design enhancement

### Pre-Requsities
- Python 3.9 (Python 3.7/3.8 can work in some cases)

### Steps to run Code
- Clone the repository
```
git clone https://github.com/Latrobe-Innovation-Hub/object-tracking-zones.git
```

- Goto the cloned folder.
```
cd yolov5-object-tracking
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv yolov5objtracking
source yolov5objtracking/bin/activate

### For Window Users
python3 -m venv yolov5objtracking
yolov5objtracking\Scripts\activate.bat
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

- Install Cuda-PyTorch binaries for using GPU devices
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Run the code with mentioned command below.
```
NOTE: don't forget to also load the appropriate .yaml file for a given model weight.pt file
      these can be found in the .\model and .\model\hub directories

# for detection only
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4"

# for detection of specific class (person)
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4" --classes 0

# for object detection + object tracking
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4"

# for object detection + object tracking + object blurring
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4" --blur-obj

# for object detection + object tracking + object blurring + different color for every bounding box
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4" --blur-obj --color-box

# for object detection + object tracking of specific class (person)
python obj_det_and_trk_zones.py --weights yolov5s.pt --source "vidoes/crowd-1.mp4" --classes 0

# for object detection + object tracking of specific class (person), from webcam, with confidence set, and load and save zones to/from file
python obj_det_and_trk_zones.py --weights yolov5n.pt --source 0 --classes 0 --conf-thres .25 --load-zones zone_save_test.txt --save-zones zone_save_test.txt

# for object detection + tracking from webcam, with confidence set per class person 75% and table? 40%, and absolute confidence (lowest possible) set at 25%
python obj_det_and_trk_zones.py --weights yolov5s.pt --source 0 --classes 0 56 --conf-thres-per-class 0.75 0.40 --conf-thres 0.25

# for black background with dots use 'obj_det_and_zones-blacked.py':
python obj_det_and_zones-blacked.py --weights yolov5s.pt --source "videos/crowd-1.mp4" --blur-obj --color-box

# to run streamlit web demo (may have requirements that need to be manually configured at this point; read console errors if so...):
# Note 1: On my Windows system pip install streamlit required python /script dir being added to path: 'set PATH=%PATH%;C:\Users\andy\AppData\Roaming\Python\Python39\Scripts\'
# Note 2: RTSP options only work if on La Trobe DIH JG (formally TC) building's network AND the Meraki cameras RTSP feed is enabled
# to exit streamlit 'ctrl-c'
streamlit run app.py
```

### Streamlit Dashboard
- If you want to run detection on streamlit app (Dashboard), you can use mentioned command below.

<b>Note:</b> Make sure, to add video in the <b>yolov5-object-tracking</b> folder, that you want to run on streamlit dashboard. Otherwise streamlit server will through an error.  Also, streamlit runs on obj_det_and_trk.py not obj_det_and_trk_zones.py
```
python -m streamlit run app.py
```

### Issues
UPDATE: have hard-coded an automatated try/fail for most popular opencv backends, so should find most suitable on its own.  However, if it is unable to, it will run an extreme test through all possible backends, which may take upto a minute to complete. So if the script starts and appears to 'hang' just wait as it is most likely doing a long search for a working video backend.

If source won't display in opencv, check utils/dataloaders.py and set the video capture backend to one suitable for the system:
```python
# Start the video capture using the default camera (index 0) with the DirectShow backend.
# If you encounter issues with the video capture, you can try different backend options by changing the second argument:
# - cv2.CAP_DSHOW: DirectShow (Windows only)
# - cv2.CAP_V4L2: Video4Linux2 (Linux only)
# - cv2.CAP_AVFOUNDATION: AVFoundation (macOS only)
cap = cv2.VideoCapture(s, cv2.CAP_DSHOW)
```

### Screenshots
<table>
  <tr>
    <td>YOLOv5 Object Detection</td>
    <td>YOLOv5 Object Tracking</td>
    <td>YOLOv5 Object Tracking + Object Blurring</td>
    <td>YOLOv5 Streamlit Dashboard</td>
  </tr>
  <tr>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525324-9aaf4b60-9336-41c3-8a27-8722bb7da731.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525332-1e84b4d5-ae4e-4c1b-9498-0ec1d4ad4bd7.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525328-f85ef474-e964-4d79-8f75-78ad4e5397d4.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525342-8d4d81f4-5e3a-45aa-9972-5f5de1c72159.png"></td>
  </tr>
 </table>

### References
 - https://github.com/ultralytics/yolov5
 - https://github.com/abewley/sort
