import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize the detector with the specified thresholds
detector = MTCNN(steps_threshold=[.2, .5, .5])

# Load the input image
image = cv2.imread('../images/family.jpg')
if image is None:
    raise FileNotFoundError("The face image file was not found. Please check the file path.")

# Detect faces in the image
result = detector.detect_faces(image)

# Check if any faces are detected
if result:
    # Load the mask image
    mask = cv2.imread('../images/mask.png', cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError("The mask image file was not found. Please check the file path.")

    # Separate the mask's alpha channel
    mask_bgr = mask[..., :3]
    mask_alpha = mask[..., 3]

    for face in result:
        # Get the bounding box of the face
        x, y, width, height = face['box'] 
        # Get the keypoints of the face
        keypoints = face['keypoints']

        # Calculate the width and height of the mask
        face_width = width
        face_height = height
        mask_width = int(1.5 * face_width)
        mask_height = int(0.8 * face_height)

        # Resize the mask to fit the face
        resized_mask_bgr = cv2.resize(mask_bgr, (mask_width, mask_height))
        resized_mask_alpha = cv2.resize(mask_alpha, (mask_width, mask_height))

        # Determine the position to place the mask
        y1, y2 = int(y + 0.2 * face_height), int(y + 0.8 * face_height)
        x1, x2 = int(x - 0.25 * face_width), int(x + 1.25 * face_width)

        # Ensure the coordinates are within the image boundaries
        y1 = max(y1, 0)
        y2 = min(y2, image.shape[0])
        x1 = max(x1, 0)
        x2 = min(x2, image.shape[1])

        # Get the region of interest on the image
        roi = image[y1:y2, x1:x2]

        # Resize the mask to match the size of roi
        resized_mask_bgr = cv2.resize(resized_mask_bgr, (roi.shape[1], roi.shape[0]))
        resized_mask_alpha = cv2.resize(resized_mask_alpha, (roi.shape[1], roi.shape[0]))

        # Blend the mask with the region of interest
        alpha_mask = resized_mask_alpha / 255.0
        for c in range(0, 3):
            roi[..., c] = roi[..., c] * (1 - alpha_mask) + resized_mask_bgr[..., c] * alpha_mask

        # Place the blended image back into the original image
        image[y1:y2, x1:x2] = roi

    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("People with Masks")
    plt.show()

else:
    print("No faces detected in the image.")
