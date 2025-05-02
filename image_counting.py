import ultralytics
import cv2
from ultralytics import solutions
import os

import os

# Get the current working directory
current_path = os.getcwd()

image_name = "my_image.jpg"  
path = os.path.join(current_path, image_name)
img = cv2.imread(path)


counter = solutions.ObjectCounter(
    show=False,  # display the output
    #region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[2],  # 0 for persons # count specific classes i.e. person and car with COCO pretrained model.
    tracker="botsort.yaml"  # choose trackers i.e "bytetrack.yaml"
)

unique_vehicle_ids = set() # to store unique vehicle IDs


results = counter.process(img)
track_ids = counter.track_ids

if results and track_ids is not None:
    for track_id in track_ids:
        unique_vehicle_ids.add(int(track_id))

# Draw the current count
current_count = len(track_ids) if results and track_ids is not None else 0
total_count = len(unique_vehicle_ids)

cv2.putText(results.plot_im, f"Currently Detected: {current_count}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(results.plot_im, f"Total Vehicles So Far: {total_count}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow('Processed Frame', results.plot_im)

while True:
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press
    if key == ord('p'):          # close window if 'p' is pressed
        break
    
cv2.destroyAllWindows()
