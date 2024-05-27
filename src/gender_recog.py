from deepface import DeepFace
import cv2
import os
from mtcnn import MTCNN

def analyze_gender(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Initialize MTCNN
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return

    print(f"Detected {len(faces)} faces in {image_path}")

    # Analyze each detected face for gender
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        face_img = img[y:y+height, x:x+width]

        try:
            result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
            if result and isinstance(result, list) and 'dominant_gender' in result[0]:
                gender = result[0]['dominant_gender']
                print(f"Face {i} - Gender: {gender}")
            else:
                print(f"Face {i} - No gender detected")
        except Exception as e:
            print(f"Error analyzing face {i}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    image_folder = "../images"
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        print(f"Analyzing {image_name}...")
        analyze_gender(image_path)
