document.addEventListener('DOMContentLoaded', function() {
  // Get references to elements
  const uploadButton = document.getElementById('upload');
  const fileInput = document.getElementById('file-input');
  const videoPreview = document.getElementById('video-preview');
  const fileName = document.getElementById('file-name');
  const buttonsContainer = document.getElementById('buttons-container');
  let uploadedFile;

  // When the "Select from device" button is clicked, reset the file input value and trigger click
  uploadButton.addEventListener('click', function() {
    fileInput.value = '';  // Reset the input value to ensure it triggers the change event
    fileInput.click();     // Open the file input dialog
  });

  // When a file is selected, display the file name, size, and video preview
  fileInput.addEventListener('change', function () {
    const selectedFile = fileInput.files[0];
    if (selectedFile) {
      // Hide upload-related elements
      document.getElementById('file-logo').style.display = 'none';
      document.getElementById('container-text').style.display = 'none';
      uploadButton.style.display = 'none';

      // Show the video preview
      const fileURL = URL.createObjectURL(selectedFile);
      videoPreview.style.display = 'block';
      videoPreview.src = fileURL; // Set the video source to the selected file

      // Display the selected file name
      fileName.textContent = `Selected video: ${selectedFile.name}`;

      // Display the file size
      const fileSize = selectedFile.size; // Size in bytes
      const formattedSize = formatFileSize(fileSize); // Format size to KB or MB
      const sizeText = `Size: ${formattedSize}`;
      const fileSizeElement = document.createElement('p');
      fileSizeElement.textContent = sizeText;
      fileName.appendChild(fileSizeElement); // Append the file size below the file name

      // Show the buttons below the video
      buttonsContainer.style.display = 'flex';

      uploadedFile = selectedFile; // Save the selected file for later use
    } else {
      // Reset everything if no file is selected
      document.getElementById('file-logo').style.display = 'block';
      document.getElementById('container-text').style.display = 'block';
      uploadButton.style.display = 'flex';
      videoPreview.style.display = 'none';
      fileName.textContent = 'No file selected';
    }
  });

  // Function to format file size in KB or MB
  function formatFileSize(sizeInBytes) {
    if (sizeInBytes < 1024) {
      return sizeInBytes + ' bytes';
    } else if (sizeInBytes < 1024 * 1024) {
      return (sizeInBytes / 1024).toFixed(2) + ' KB';
    } else {
      return (sizeInBytes / (1024 * 1024)).toFixed(2) + ' MB';
    }
  }
});

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle('open');
}
// Function to close the sidebar by removing the 'open' class
function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.remove('open'); // Removes the 'open' class, closing the sidebar
}


//BUTTON 2

// Frontend JavaScript to handle video upload and object detection
videoObjectDetectionButton.addEventListener('click', function() {
  if (uploadedFile) {
    const formData = new FormData();
    formData.append('video', uploadedFile);  // Append the video file to the form data

    // Display processing message
    alert('Processing the video with Object Detection...');

    // Send the video to the backend for processing
    fetch('/upload_video', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        alert('Processing completed!');

        // Show the processed video and download button
        processedVideoContainer.style.display = 'block';
        processedVideo.src = data.output_video_url;  // Set the processed video URL
        downloadButton.style.display = 'block';
        downloadButton.href = data.output_video_url;  // Set the download link
      } else {
        alert('Error processing video: ' + data.message);
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('There was an error processing the video.');
    });
  } else {
    alert('Please upload a video first!');
  }
});











