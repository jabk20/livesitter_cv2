import cv2
import numpy as np
from pytube import YouTube

# Function to detect people in a frame using YOLOv3
def detect_people(frame, net, ln):
    personIdx = 0  # Assuming the class ID for "person" is 0 in your YOLO model
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten(): # type: ignore
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x + w, y + h), classIDs[i])
            results.append(r)

    return results

# Specify the paths to the YOLOv3 model weights and configuration files
weights_path = "D:/chrome download/yolov3-tiny.weights"
config_path = "D:/chrome download/yolov3-tiny.cfg"

# YouTube video URL
youtube_video_url = "https://youtu.be/kUEwTXSOY0c"

# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Get the output layer names of the network for YOLOv3
ln = net.getUnconnectedOutLayersNames()

# Open the YouTube video
try:
    yt = YouTube(youtube_video_url)
    video_stream = yt.streams.get_highest_resolution()

    if video_stream:
        video_stream.download(filename="downloaded_video.mp4")
    else:
        print("No suitable stream found for download.")
        exit(1)

    # Open the downloaded video
    cap = cv2.VideoCapture("downloaded_video.mp4")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection on the frame
        results = detect_people(frame, net, ln)

        # Draw bounding boxes around the detected people
        for (confidence, bbox, classID) in results:
            (startX, startY, endX, endY) = bbox
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Release the video file
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")
