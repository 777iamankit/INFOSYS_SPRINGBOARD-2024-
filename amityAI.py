import cv2
import numpy as np

# Replace with the URL of your phone's camera stream
VIDEO_STREAM_URL = "http://192.0.0.4:8080"

# Paths to YOLO files
YOLO_WEIGHTS = "dnn_model/yolov4.weights"  # Replace with the path to your YOLO weights file
YOLO_CONFIG = "dnn_model/yolov4.cfg"       # Replace with the path to your YOLO config file
COCO_NAMES = "dnn_model/classes.txt"        # Replace with the path to your COCO class names file

def load_yolo():
    # Load YOLO model
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    # Set backend and target for faster inference (use CUDA if supported)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Change to DNN_TARGET_CUDA for GPU
    # Load class names
    with open(COCO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def get_outputs(net):
    # Get all layer names
    layer_names = net.getLayerNames()
    
    # Ensure compatibility with older OpenCV versions
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        # If getUnconnectedOutLayers() returns scalar values
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame, net, output_layers, classes):
    height, width, _ = frame.shape

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # Perform forward pass
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Parse the outputs
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Filter detections by confidence
                # Scale bounding box back to frame size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        detected_objects.append((classes[class_ids[i]], confidences[i], (x, y, w, h)))

    return detected_objects

def main():
    # Load YOLO model
    net, classes = load_yolo()
    output_layers = get_outputs(net)

    # Capture video from the phone camera stream
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)

    if not cap.isOpened():
        print("Error: Unable to access the video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect objects in the frame
        detected_objects = detect_objects(frame, net, output_layers, classes)

        # Draw bounding boxes and display object counts
        object_count = {}
        for obj_class, confidence, bbox in detected_objects:
            x, y, w, h = bbox
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label the object
            label = f"{obj_class} {int(confidence * 100)}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Count objects
            object_count[obj_class] = object_count.get(obj_class, 0) + 1

        # Display the object count
        for i, (obj_class, count) in enumerate(object_count.items()):
            cv2.putText(frame, f"{obj_class}: {count}", (10, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the video
        cv2.imshow("YOLO Object Counter", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
