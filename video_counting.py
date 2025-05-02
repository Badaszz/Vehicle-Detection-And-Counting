import ultralytics
import cv2
from ultralytics import solutions
import os


# Get the current working directory
current_path = os.getcwd()
video_name = "short_clip.mp4" #replace with your image name
#path = os.path.join(current_path, video_name)
path = "G:\Other computers\My Laptop\projects\Plant Diseases\Vehicle_detection_and_counting\short_clip.mp4"
cap = cv2.VideoCapture(path)

# Video writer to create output video file
# Uncomment the following lines to save the output video
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=False,  # display the output, change to True if you want to see the output in a window
    #region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[2],   # count specific classes  # 0 for persons
    tracker="botsort.yaml"  # choose trackers i.e "bytetrack.yaml"
)

unique_vehicle_ids = set() # to store unique vehicle IDs

# Process video
while True:
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.") #when the video ends
        break

    results = counter.process(im0) # process the frame using the object counter
    track_ids = counter.track_ids # get the track IDs of detected vehicles

    if results and track_ids is not None:
        for track_id in track_ids:
            unique_vehicle_ids.add(int(track_id)) # add unique track IDs to the set

    
    current_count = len(track_ids) if results and track_ids is not None else 0 # current count of vehicles
    total_count = len(unique_vehicle_ids) # total count of unique vehicles

    cv2.putText(results.plot_im, f"Currently Detected: {current_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # write the current count
    cv2.putText(results.plot_im, f"Total Vehicles So Far: {total_count}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)# write the total count
    
    cv2.imshow('Processed Frame', results.plot_im)
    
    # Uncomment the following line to save the output video
    # video_writer.write(results.plot_im)  # write the processed frame.
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

print(f"Total unique vehicles detected: {len(unique_vehicle_ids)}")
#print("Processing complete. Output saved as 'object_counting_output.avi'.")

cap.release()
#video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows

