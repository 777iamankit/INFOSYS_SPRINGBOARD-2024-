from imageai.Detection import VideoObjectDetection
import os
import time
import sys

# Get input and output file paths from command line arguments
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Start timing
start_time = time.time()

# Set the execution path
execution_path = os.getcwd()

# Ensure the output directory exists
output_folder = os.path.dirname(output_file_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the output file path with a filename
# output_file_path = os.path.join(output_folder, "processed_video.mp4")  # This is passed from Flask now

# Reduce unnecessary logging: Only print every 5 seconds for frame-level data
def forFrame(frame_number, output_array, output_count):
    # Print data every 5th frame (you can adjust this as needed)
    if frame_number % 5 == 0:
        print(f"Frame {frame_number}: {output_count} unique objects detected.")
        print("Detected Objects:", output_array)

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    # Print data every 30 seconds
    if second_number % 30 == 0:
        print(f"Second {second_number}: {average_output_count} objects detected on average.")
        print("Object Count per Frame:", count_arrays)

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    # Print data every minute
    print(f"Minute {minute_number}: Average object count per second is {average_output_count}")
    print("Detected Objects per Frame:", output_arrays)

# Initialize the video detector
video_detector = VideoObjectDetection()
video_detector.setModelTypeAsTinyYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "models/tiny-yolov3.pt"))
video_detector.loadModel()

# Process the video and save the output
video_detector.detectObjectsFromVideo(
    input_file_path=input_file_path,
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
