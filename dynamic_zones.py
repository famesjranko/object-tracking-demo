"""
Author: Andrew J McDonald
Date: 17.07.2023
"""

import cv2
import string
import random

# Define the video capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the global variables
drawing = False
x, y, w, h = 0, 0, 0, 0

# Define an empty list to hold the dynamic zones
zone_rois = []
zone_counter = 0

# Define a mouse callback function to capture the mouse events
def draw_rectangle_old(event, x_new, y_new, flags, params):
    global x, y, w, h, drawing
    # Capture the mouse down event and initialize the rectangle drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = x_new, y_new
        
    # Capture the mouse up event and finalize the rectangle drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x_new - x, y_new - y
        
    # Capture the mouse move event and update the rectangle dimensions
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x_new - x, y_new - y

def draw_rectangle_working(event, x_new, y_new, flags, params):
    global x, y, w, h, drawing, zone_rois
    
    # Capture the mouse down event and initialize the rectangle drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = x_new, y_new
        w, h = 0, 0
        
    # Capture the mouse up event and finalize the rectangle drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x_new - x, y_new - y
        if w >= 20 and h >= 20:
            zone = {'name': f'Zone {len(zone_rois)+1}', 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
            zone_rois.append(zone)
            print(f'Added {zone["name"]}: {zone["roi"]}')
        
    # Capture the mouse move event and update the rectangle dimensions
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x_new - x, y_new - y
    
    # Capture the right mouse button down event and remove the zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Use a list comprehension to filter the list of zones and remove the correct one
        #zone_rois = [zone for zone in zone_rois if not (x_new > zone['roi'][0] and x_new < zone['roi'][0] + zone['roi'][2] and y_new > zone['roi'][1] and y_new < zone['roi'][1] + zone['roi'][3])]
        # Calculate the distance between the center of each zone and the current mouse position
        distances = [(zone['roi'][0] + zone['roi'][2] / 2 - x_new) ** 2 + (zone['roi'][1] + zone['roi'][3] / 2 - y_new) ** 2 for zone in zone_rois]
        # Find the index of the zone with the smallest distance
        closest_index = distances.index(min(distances))
        # Remove the zone with the smallest distance
        
        print(f'Removed {zone_rois[closest_index]["name"]}: {zone_rois[closest_index]["roi"]}')
        del zone_rois[closest_index]

def draw_rectangle(event, x_new, y_new, flags, params):
    global x, y, w, h, drawing, zone_rois, zone_counter
    
    # Capture the mouse down event and initialize the rectangle drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x, y = x_new, y_new
        w, h = 0, 0
        
    # Capture the mouse up event and finalize the rectangle drawing
    #elif event == cv2.EVENT_LBUTTONUP:
    #    drawing = False
    #    w, h = x_new - x, y_new - y
    #    if w >= 20 and h >= 20:
    #        zone = {'name': f'Zone {len(zone_rois)+1}', 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
    #        zone_rois.append(zone)
    #        print(f'Added {zone["name"]}: {zone["roi"]}')
            
    # Capture the mouse up event and finalize the rectangle drawing
    #elif event == cv2.EVENT_LBUTTONUP:
    #    drawing = False
    #    w, h = x_new - x, y_new - y
    #    if w >= 20 and h >= 20:
    #        zone_name = f'Zone {zone_counter}'
    #        while any(zone['name'] == zone_name for zone in zone_rois):
    #            zone_counter += 1
    #            zone_name = f'Zone {zone_counter}'
    #            
    #        zone = {'name': zone_name, 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
    #        zone_rois.append(zone)
    #        print(f'Added {zone["name"]}: {zone["roi"]}')
    #        zone_counter += 1
    
    # Capture the mouse up event and finalize the rectangle drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = x_new - x, y_new - y
        if w >= 20 and h >= 20:
            # Generate a unique id for the zone
            #zone_id = str(uuid.uuid4())
            # Check if the id is already used, and generate a new one if necessary
            #while zone_id in [zone['id'] for zone in zone_rois]:
            #    zone_id = str(uuid.uuid4())
            # Create the new zone dictionary
            #zone = {'id': zone_id, 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
            #zone_rois.append(zone)
            #print(f'Added Zone {zone["id"]}: {zone["roi"]}')
            
            # Generate a unique id for the zone
            zone_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
            # Check if the id is already used, and generate a new one if necessary
            while zone_id in [zone['id'] for zone in zone_rois]:
                zone_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
            # Create the new zone dictionary
            #zone = {'id': zone_id, 'name': f'Zone {len(zone_rois)+1}', 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
            zone = {'id': zone_id, 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
            zone_rois.append(zone)
            print(f'Added {zone["id"]}: {zone["roi"]}')

    # Capture the mouse move event and update the rectangle dimensions
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            w, h = x_new - x, y_new - y
    
    # Capture the right mouse button down event and remove the zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Calculate the distance between the center of each zone and the current mouse position
        distances = [(zone['roi'][0] + zone['roi'][2] / 2 - x_new) ** 2 + (zone['roi'][1] + zone['roi'][3] / 2 - y_new) ** 2 for zone in zone_rois]
        # Find the index of the zone with the smallest distance
        closest_index = distances.index(min(distances))
        # Remove the zone with the smallest distance
        print(f'Removed {zone_rois[closest_index]["id"]}: {zone_rois[closest_index]["roi"]}')
        del zone_rois[closest_index]
            
# Create a window to display the video feed
cv2.namedWindow('frame')
# Set the mouse callback function for the video feed window
cv2.setMouseCallback('frame', draw_rectangle)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw the rectangle on the video feed
    if drawing:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Create the dynamic zones based on the rectangle drawn by the user
    if not drawing and w != 0 and h != 0:
        #zone_roi = {'name': 'Zone {}'.format(len(zone_rois) + 1), 'color': (0, 0, 255), 'roi': (x, y, w, h), 'zone_dict': {'object_count': 0, 'unique_ids': set()}, 'types': []}
        #zone_rois.append(zone_roi)
        x, y, w, h = 0, 0, 0, 0
        
    # Draw the dynamic zones on the video feed
    #for zone_roi in zone_rois:
        #cv2.rectangle(frame, (zone_roi['roi'][0], zone_roi['roi'][1]), (zone_roi['roi'][0] + zone_roi['roi'][2], zone_roi['roi'][1] + zone_roi['roi'][3]), zone_roi['color'], 2)
    #    cv2.rectangle(frame, zone_roi['roi'], zone_roi['color'], 2)
    #    cv2.putText(frame, zone_roi['name'], (zone_roi['roi'][0], zone_roi['roi'][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_roi['color'], 2)
    
    # Draw the zones on the frame
    for zone in zone_rois:
        cv2.rectangle(frame, zone['roi'], zone['color'], 2)
        cv2.putText(frame, zone['id'], (zone['roi'][0] + 10, zone['roi'][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, zone['color'], 2, cv2.LINE_AA)
    
    # Display the video feed
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
