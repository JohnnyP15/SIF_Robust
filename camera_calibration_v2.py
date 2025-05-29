import numpy as np
import cv2
import glob

# Chessboard size (number of internal corners)
pattern_size = (9, 6)  # Adjust according to your chessboard

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane.

# Define criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Read images and find chessboard corners
images = glob.glob('calibration_images/*.jpg')  # Adjust the path as needed

print(f"Found {len(images)} images for calibration.")

for idx, fname in enumerate(images):
    print(f"Processing image {idx + 1}/{len(images)}: {fname}")

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        print(f"Chessboard corners found in image {idx + 1}.")
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    else:
        print(f"Chessboard corners not found in image {idx + 1}. Skipping.")

cv2.destroyAllWindows()

# Perform camera calibration
print("\nStarting camera calibration...")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

if ret:
    print("\nCamera calibration successful!")
    print("\nCamera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)

    # Save the calibration parameters to an XML file
    cv_file = cv2.FileStorage('calibration_data.xml', cv2.FILE_STORAGE_WRITE)
    cv_file.write('camera_matrix', mtx)
    cv_file.write('distortion_coefficients', dist)
    cv_file.release()

else:
    print("\nCamera calibration failed. Check your input images and try again.")

# Print the calibration results
print("\nCamera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)
