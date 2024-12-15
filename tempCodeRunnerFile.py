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