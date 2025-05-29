import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
template_path = 'Check.png'
image_path = 'undistorted_image.jpg'
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Load the original color images for display
template_color = cv2.imread(template_path)  # By default, OpenCV uses BGR
image_color = cv2.imread(image_path)
# Convert BGR to RGB for Matplotlib
template_rgb = cv2.cvtColor(template_color, cv2.COLOR_BGR2RGB)
image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

# Function to calculate the distance between points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Create a BFMatcher object with distance measurement
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors_template, descriptors_image)
matches = sorted(matches, key=lambda x: x.distance)

# Filter out matches that are too far away from their neighbors
good_matches = []
min_neighbour_dist = 12.5  # Minimum distance to consider a match as a neighbor

for i, match in enumerate(matches):
    pt_i = keypoints_image[match.trainIdx].pt
    close_neighbors = 0
    for j, other_match in enumerate(matches):
        if i == j:
            continue
        pt_j = keypoints_image[other_match.trainIdx].pt
        if euclidean_distance(pt_i, pt_j) < min_neighbour_dist:
            close_neighbors += 1

    # If the match has at least one neighbor within the minimum distance, we keep it
    if close_neighbors > 0:
        good_matches.append(match)

# Apply RANSAC to find a robust homography
if len(good_matches) >= 4:  # Homography calculation needs at least 4 matches
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Calculate the number of inliers
    num_inliers = sum(matchesMask)

    # Use the homography matrix to transform the corners from template image to observed image
    h, w = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Draw lines between the transformed corners in the scene
    image = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Print statement based on the number of inliers
    if num_inliers > 8.9:  # Threshold for a 'successful' match can be adjusted
        print("The template image was successfully found in the target image!")
    else:
        print("The template image was not found in the target image.")
else:
    matchesMask = None
    print("Not enough matches to calculate homography.")

# Draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(template_rgb, keypoints_template, image_rgb, keypoints_image, good_matches, None, **draw_params)

plt.imshow(img3), plt.show()
