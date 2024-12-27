from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from imageai.Detection import VideoObjectDetection

# Initialize Flask app
app = Flask(__name__)

# Set folder paths for uploads and outputs
execution_path = os.getcwd()
app.config['UPLOAD_FOLDER'] = os.path.join(execution_path, 'static/uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(execution_path, 'static/output')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Image processing (Imageai for object detection)
video_detector = VideoObjectDetection()
video_detector.setModelTypeAsTinyYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "static/models/tiny-yolov3.pt"))
video_detector.loadModel()

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to upload image or video
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Image Processing
            if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
                return render_template('index.html', file_uploaded=True, filename=filename)
            
            # Video Processing
            elif file.filename.lower().endswith(('mp4', 'avi', 'mov')):
                return render_template('index.html', file_uploaded=True, filename=filename)
    
    return render_template('index.html', file_uploaded=False)

# Image processing
@app.route('/image-processing/<filename>', methods=['GET'])
def process_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_cv2 = cv2.imread(filepath)
    img_mpl = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Output 1: RGB Channels
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(img_mpl[:, :, 0], cmap='Reds')
    axs[1].imshow(img_mpl[:, :, 1], cmap='Greens')
    axs[2].imshow(img_mpl[:, :, 2], cmap='Blues')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    output1_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output1.png')
    plt.savefig(output1_path)
    plt.close()

    # Output 2: Histogram
    pd.Series(img_mpl.flatten()).plot(kind='hist', bins=50, title='Pixel Distribution')
    output2_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output2.png')
    plt.savefig(output2_path)
    plt.close()

    # Output 3: BGR to RGB Conversion
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_cv2_rgb)
    ax.axis('off')
    plt.tight_layout()
    output3_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output3.png')
    plt.savefig(output3_path)
    plt.close()

    return render_template(
        'index.html',
        file_uploaded=True,
        filename=filename,
        output1='output1.png',
        output2='output2.png',
        output3='output3.png'
    )

# Video processing
@app.route('/video-processing/<filename>', methods=['GET'])
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_filename = 'processed_' + os.path.basename(video_path)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Read first frame and find corners
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        out.write(img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()

    return render_template('index.html', file_uploaded=True, filename=filename, video_output=output_filename)

#object tracking
@app.route('/object-tracking/<filename>', methods=['GET'])
def object_tracking(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error: Cannot open video file."

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        return "Error: Cannot read the video file."

    # Ensure frame is in the correct format (convert if necessary)
    frame = frame.astype(np.uint8)  # Ensure it's a valid uint8 frame for ROI selection

    # Select ROI (Uncomment for debugging)
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    
    # Uncomment for hardcoded ROI during debugging
    # bbox = (100, 100, 200, 200)  # Example: hardcoded ROI

    # Initialize the tracker (use CSRT if available)
    if hasattr(cv2, 'TrackerCSRT_create'):
        tracker = cv2.TrackerCSRT_create()
    else:
        return "Error: TrackerCSRT is not supported by your OpenCV version."

    tracker.init(frame, bbox)

    # Prepare to write output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_filename = 'tracked_' + os.path.basename(video_path)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker and draw the bounding box
        success, bbox = tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()

    return render_template(
        'index.html',
        file_uploaded=True,
        filename=filename,
        video_output=output_filename
    )


@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        return f"Error: File {filename} does not exist.", 404
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        filename, 
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)
