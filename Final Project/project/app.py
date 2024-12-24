from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess

app = Flask(__name__)

# Directories for file uploads and output
upload_folder = os.path.join(os.getcwd(), 'uploads')
output_folder = os.path.join(os.getcwd(), 'processed_videos')

# Ensure the directories exist
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded video file
    input_file_path = os.path.join(upload_folder, video_file.filename)
    video_file.save(input_file_path)

    # Define the output file path
    output_file_path = os.path.join(output_folder, f"processed_{video_file.filename}")

    # Call the external OpenCV script (process_video.py) to process the uploaded video
    try:
        # Run the external process_video.py script (existing functionality)
        subprocess.run(['python', 'process_video.py', input_file_path, output_file_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Error processing video', 'details': str(e)})

    # Return the download link for the processed video
    return jsonify({'download_link': f'/download/{os.path.basename(output_file_path)}'})






@app.route('/video_object_tracking', methods=['POST'])
def video_object_tracking():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded video file
    input_file_path = os.path.join(upload_folder, video_file.filename)
    video_file.save(input_file_path)

    # Define the output file path for object tracking processed video
    output_file_path = os.path.join(output_folder, f"tracked_{video_file.filename}")

    # Call the external object tracking script (object_tracking.py) to process the uploaded video
    try:
        # Run the object_tracking.py script (new functionality)
        subprocess.run(['python', 'object_tracking.py', input_file_path, output_file_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Error processing video with object tracking', 'details': str(e)})

    # Return the download link for the processed video with object tracking
    return jsonify({'download_link': f'/download/{os.path.basename(output_file_path)}'})






@app.route('/download/<filename>')
def download_video(filename):
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
