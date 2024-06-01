
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

def overlay_sunglasses(image, face):
    sunglasses = cv2.imread('../images/glasses.jpg', cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError("The sunglasses image file was not found. Please check the file path.")

    x, y, width, height = face['box']
    keypoints = face['keypoints']

    eye_width = abs(keypoints['right_eye'][0] - keypoints['left_eye'][0])
    glasses_width = int(eye_width * 2)
    glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

    resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

    y1, y2 = keypoints['left_eye'][1] - int(glasses_height / 2), keypoints['left_eye'][1] + int(glasses_height / 2)
    x1, x2 = keypoints['left_eye'][0] - int(glasses_width / 4), keypoints['left_eye'][0] + int(3 * glasses_width / 4)

    y1 = max(y1, 0)
    y2 = min(y2, image.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, image.shape[1])

    roi = image[y1:y2, x1:x2]

    bg_sunglasses = resized_sunglasses[..., :3]
    mask = resized_sunglasses[..., 3:] / 255.0

    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
    bg_sunglasses = cv2.resize(bg_sunglasses, (roi.shape[1], roi.shape[0]))

    for c in range(0, 3):
        roi[..., c] = roi[..., c] * (1 - mask) + bg_sunglasses[..., c] * mask

    image[y1:y2, x1:x2] = roi
    return image
