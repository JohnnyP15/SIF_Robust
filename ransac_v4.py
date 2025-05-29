import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path, color=False):
    """Load an image in grayscale or color."""
    if color:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# Load images
template_path = 'Check.png'
image_path = 'cluster.jpg'
template = load_image(template_path)
image = load_image(image_path)
template_rgb = load_image(template_path, color=True)
image_rgb = load_image(image_path, color=True)

# Check if images were loaded properly
if template is None or image is None:
    print("Error: One or both of the images couldn't be loaded.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

# Visualize Keypoints
img_keypoints_template = cv2.drawKeypoints(template_rgb, keypoints_template, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_keypoints_image = cv2.drawKeypoints(image_rgb, keypoints_image, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.imshow(img_keypoints_template)
plt.title('Keypoints in Template')
plt.subplot(1, 2, 2)
plt.imshow(img_keypoints_image)
plt.title('Keypoints in Image')
plt.show()

# Create a BFMatcher object with distance measurement
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_template, descriptors_image, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < .75 * n.distance:
        good_matches.append(m)

# Visualize Matches
img_matches = cv2.drawMatches(template_rgb, keypoints_template, image_rgb, keypoints_image, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title('Top 10 Matches')
plt.show()

# Apply RANSAC to find a robust homography
if len(good_matches) >= 4:
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None and M.shape == (3, 3):
        matchesMask = mask.ravel().tolist()
        num_inliers = sum(matchesMask)

        if num_inliers > 8.9:
            print("The template image was successfully found in the target image!")
        else:
            print("The template image was not found in the target image.")

        # Use the homography matrix to transform the corners from template image to observed image
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        image_rgb = cv2.polylines(image_rgb, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        matchesMask = None
        print("Homography computation failed.")
else:
    matchesMask = None
    print("Not enough matches to calculate homography.")

# Draw matches
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img3 = cv2.drawMatches(template_rgb, keypoints_template, image_rgb, keypoints_image, good_matches, None, **draw_params)

# Display results
plt.figure(figsize=(15, 10))
plt.imshow(img3)
plt.title('Template Matching Results')
plt.show()
