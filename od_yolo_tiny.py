from imageai.Detection import VideoObjectDetection
import os
import time
import torch

# Start timing
start_time = time.time()

# Set the execution path
execution_path = os.getcwd()

# Ensure the output directory exists
output_folder = os.path.join(execution_path, "traffic_detected_yolo_tiny")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the output file path with a filename
output_file_path = os.path.join(output_folder, "processed_video.mp4")

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

# Initialize the video detector
video_detector = VideoObjectDetection()
video_detector.setModelTypeAsTinyYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "models/tiny-yolov3.pt"))
video_detector.loadModel()

# Process the video and save the output
video_detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "videos/video1.mp4"),
    output_file_path=output_file_path,  # Save output to the specified folder and filename
    frames_per_second=10,
    per_second_function=forSeconds,
    per_frame_function=forFrame,
    per_minute_function=forMinute,
    minimum_percentage_probability=30
)

# End timing and calculate the duration
end_time = time.time()
execution_duration = end_time - start_time

# Print the time taken to process the video
print("Time taken to run the code:", execution_duration, "seconds")
