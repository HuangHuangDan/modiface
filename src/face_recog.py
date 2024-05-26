import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

def visualize_faces(image, faces, title):
    for face in faces:
        box = face['box']
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def main():
    # Initialize the MTCNN detector
    detector = MTCNN(steps_threshold=[.2, .5, .5])

    # Load the input image
    image_path = "../images/family.jpg"  # Adjust the path to your image file
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image file at {image_path} was not found. Please check the file path.")

    # Detect faces in the image
    faces = detector.detect_faces(image)
    print(f"Detected {len(faces)} faces")

    # Visualize the detected faces
    visualize_faces(image, faces, "Detected Faces")

if __name__ == '__main__':
    main()
