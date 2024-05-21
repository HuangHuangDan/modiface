import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Initialize the detector with the specified thresholds
detector = MTCNN(steps_threshold=[.2, .7, .7])

# Load the input image
image = cv2.imread('/Users/torryy/Documents/modiface/images/family.jpg')
if image is None:
    raise FileNotFoundError("The face image file was not found. Please check the file path.")

# Detect faces in the image
result = detector.detect_faces(image)

# Check if any faces are detected
if result:
    # Load the sunglasses image
    sunglasses = cv2.imread('/Users/torryy/Documents/modiface/images/glasses.jpg', cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError("The sunglasses image file was not found. Please check the file path.")

    for face in result:
        # Get the bounding box of the face
        x, y, width, height = face['box'] 
        # Get the keypoints of the face
        keypoints = face['keypoints']

        # Calculate the width and height of the sunglasses
        eye_width = abs(keypoints['right_eye'][0] - keypoints['left_eye'][0])
        glasses_width = int(eye_width * 2)
        glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

        # Resize the sunglasses to fit the face
        resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

        # Determine the position to place the sunglasses
        y1, y2 = keypoints['left_eye'][1] - int(glasses_height / 2), keypoints['left_eye'][1] + int(glasses_height / 2)
        x1, x2 = keypoints['left_eye'][0] - int(glasses_width / 4), keypoints['left_eye'][0] + int(3 * glasses_width / 4)

        # Ensure the coordinates are within the image boundaries
        y1 = max(y1, 0)
        y2 = min(y2, image.shape[0])
        x1 = max(x1, 0)
        x2 = min(x2, image.shape[1])

        # Get the region of interest on the image
        roi = image[y1:y2, x1:x2]

        # Convert the sunglasses image to BGR format and extract the alpha mask
        bg_sunglasses = resized_sunglasses[..., :3]
        mask = resized_sunglasses[..., 3:] / 255.0

        # Resize the mask to match the roi dimensions
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
        bg_sunglasses = cv2.resize(bg_sunglasses, (roi.shape[1], roi.shape[0]))

        # Blend the sunglasses with the region of interest
        for c in range(0, 3):
            roi[..., c] = roi[..., c] * (1 - mask) + bg_sunglasses[..., c] * mask

        # Place the blended image back into the original image
        image[y1:y2, x1:x2] = roi

    # Display the result
    cv2.imshow('With Sunglasses', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces detected in the image.")


import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

# Initialize the detector with the specified thresholds
detector = MTCNN(steps_threshold=[.2, .7, .7])

# Load the image
image = cv2.imread('/Users/torryy/Documents/modiface/images/family.jpg')
if image is None:
    raise FileNotFoundError("The face image file was not found. Please check the file path.")

# Detect faces in the image
result = detector.detect_faces(image)

# Check if any faces are detected
if result:
    # Load the sunglasses image
    sunglasses = cv2.imread('/Users/torryy/Documents/modiface/images/glasses.jpg', cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError("The sunglasses image file was not found. Please check the file path.")

    for face in result:
        # Get the bounding box of the face
        x, y, width, height = face['box']
        # Get the keypoints of the face
        keypoints = face['keypoints']

        # Calculate the width and height of the sunglasses
        eye_width = abs(keypoints['right_eye'][0] - keypoints['left_eye'][0])
        glasses_width = int(eye_width * 2)
        glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

        # Resize the sunglasses to fit the face
        try:
            resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))
        except Exception as e:
            print(f"Error resizing sunglasses: {e}")
            continue

        # Determine the position to place the sunglasses
        y1, y2 = keypoints['left_eye'][1] - int(glasses_height / 2), keypoints['left_eye'][1] + int(glasses_height / 2)
        x1, x2 = keypoints['left_eye'][0] - int(glasses_width / 4), keypoints['left_eye'][0] + int(3 * glasses_width / 4)

        # Ensure the coordinates are within the image boundaries
        y1 = max(y1, 0)
        y2 = min(y2, image.shape[0])
        x1 = max(x1, 0)
        x2 = min(x2, image.shape[1])

        # Get the region of interest on the image
        roi = image[y1:y2, x1:x2]

        # Ensure the ROI and sunglasses have compatible dimensions
        try:
            if roi.shape[0] != glasses_height or roi.shape[1] != glasses_width:
                glasses_height, glasses_width = roi.shape[0], roi.shape[1]
                resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))
        except Exception as e:
            print(f"Error resizing sunglasses to match ROI: {e}")
            continue

        # Convert the sunglasses image to BGR format and extract the alpha mask
        bg_sunglasses = resized_sunglasses[..., :3]
        mask = resized_sunglasses[..., 3:] / 255.0

        # Resize the mask to match the roi dimensions
        try:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
            bg_sunglasses = cv2.resize(bg_sunglasses, (roi.shape[1], roi.shape[0]))
        except Exception as e:
            print(f"Error resizing mask or bg_sunglasses: {e}")
            continue

        # Blend the sunglasses with the region of interest
        try:
            for c in range(0, 3):
                roi[..., c] = roi[..., c] * (1 - mask) + bg_sunglasses[..., c] * mask
        except Exception as e:
            print(f"Error blending sunglasses with ROI: {e}")
            continue

        # Place the blended image back into the original image
        image[y1:y2, x1:x2] = roi

    # Display the result
    cv2.imshow('With Sunglasses', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

else:
    print("No faces detected in the image.")
