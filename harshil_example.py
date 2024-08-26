"""
Author: Andrew J McDonald
Date: 17.07.2023
"""

## simple example of how to import the obj_det_and_trk_zones2 file and use detect()

import obj_det_and_trk_zones_importable as dt
from pathlib import Path
import cv2

ROOT = '.'

# set the input image location 
# default is ROOT dir
# example: "/images/img.jpg" will be in the image dir in the project ROOT directory
input = 'test.jpg'

# run detection on source image and save to var
output = dt.detect(source=input)

# create output directory if it doesn't exist
output_dir = Path(ROOT) / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

# save output image to ROOT/output/"filename.ext"
cv2.imwrite(str(output_dir / input), output)

# display image to user with opencv
cv2.imshow('frame', output)

cv2.waitKey(0) # Wait indefinitely until a key is pressed
cv2.destroyAllWindows() # Close the window when a key is pressed
