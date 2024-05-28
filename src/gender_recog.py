from deepface import DeepFace
import cv2
from mtcnn import MTCNN
from add_mask import overlay_mask
from add_sunglasses import overlay_sunglasses
import matplotlib.pyplot as plt

def analyze_and_modify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    detector = MTCNN()
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return

    print(f"Detected {len(faces)} faces in {image_path}")

    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        face_img = img[y:y+height, x:x+width]

        try:
            result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
            if result and isinstance(result, list) and 'dominant_gender' in result[0]:
                gender = result[0]['dominant_gender']
                print(f"Face {i} - Gender: {gender}")
                if gender == 'Man':
                    img = overlay_sunglasses(img, face)
                elif gender == 'Woman':
                    img = overlay_mask(img, face)
            else:
                print(f"Face {i} - No gender detected")
        except Exception as e:
            print(f"Error analyzing face {i}: {str(e)}")

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Modified Image")
    plt.show()

if __name__ == "__main__":
    image_folder = "../images"
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        print(f"Analyzing {image_name}...")
        analyze_and_modify_image(image_path)
