import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image you want to undistort
img_path = 'calibration_image_3.jpg.jpg'
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (640, 480))

# Load calibration data
cv_file = cv2.FileStorage('calibration_data_Backup.xml', cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('camera_matrix').mat()
dist_coeffs = cv_file.getNode('distortion_coefficients').mat()
cv_file.release()

# Undistort the image
undistorted_img = cv2.undistort(img_resized, camera_matrix, dist_coeffs)

# Display the original and undistorted images using matplotlib
plt.subplot(121), plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)), plt.title('Undistorted')
plt.show()

# Save the undistorted image
cv2.imwrite('undistorted_image.jpg', undistorted_img)
