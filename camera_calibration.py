import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
CHECKERBOARD = (9, 6)  # Ensure this matches your checkerboard

# Criteria for termination of the iterative algorithm used to refine corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Load images
images = glob.glob('calibration_images/*.jpg')  # Update the path to your images

valid_image_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)  # Show each image for 100 ms
        valid_image_count += 1

cv2.destroyAllWindows()

# Check if sufficient valid images were found
if valid_image_count < 10:
    print("Not enough valid images found for calibration. Please capture more images.")
else:
    # Standard calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("RMS:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Save the calibration results
    np.savez('calibration_data_standard.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print("Camera calibration is completed.")
