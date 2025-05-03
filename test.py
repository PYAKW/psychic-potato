from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# Load your trained model
model = YOLO("runs\detect/train5\weights/best.pt")  # update this path if needed

# Load the video file
video_path = "spark s3.mp4"  # replace with your actual video file path
cap = cv2.VideoCapture(video_path)

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Show the frame in a window
    cv2.imshow("YOLO Detection", annotated_frame)

    # Add this line — required for the window to update
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Clean up
cap.release()
cv2.destroyAllWindows()
