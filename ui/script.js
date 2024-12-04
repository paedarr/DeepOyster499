// Get the input element
const imageUpload = document.getElementById('imageUpload');

// Add an event listener to detect file upload
imageUpload.addEventListener('change', function() {
  // Get the selected file
   const file = imageUpload.files[0];

  // Create a FileReader object
  const reader = new FileReader();

  // Set up the reader's onload event handler
  reader.onload = function(e) {
    // Get the image data URL
    const imageDataUrl = e.target.result;

    // Display the uploaded image
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = imageDataUrl;
  };

  // Read the selected file as Data URL
  reader.readAsDataURL(file);
});