import cv2
import matplotlib.pyplot as plt
import pytesseract

# Path to the Tesseract executable (change this according to your system)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load an image from file
image_path = 'C:\\Users\\Sonia K\\OneDrive\\Documents\\My Work\\image-processing-opencv\\images\\test_image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform Otsu's thresholding
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the processed image
plt.imshow(thresholded_image, cmap='gray')
plt.title('Processed Image')
plt.show()

# Display the thresholded image for inspection
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Perform OCR (text extraction)
extracted_text = pytesseract.image_to_string(thresholded_image)

# Verify if any text is extracted
if extracted_text:
    # Display the extracted text
    print("Extracted Text:")
    print(extracted_text)
    # Save extracted text to a .txt file
    output_file_path = 'C:\\Users\\Sonia K\\OneDrive\\Documents\\My Work\\image-processing-opencv\\recognized.txt'
    with open(output_file_path, 'w') as file:
        file.write(extracted_text)
    print(f"Extracted text saved to '{output_file_path}'")
else:
    print("No text detected. OCR may have failed to recognize text in the image.")

