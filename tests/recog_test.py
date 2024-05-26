import unittest
import cv2 
from mtcnn import MTCNN
import matplotlib.pyplot as plt

mtcnn = None

class TestMTCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mtcnn
        mtcnn = MTCNN()

    def test_detect_faces(self):
        """
        Detect faces and landmarks on an image
        """
        testpic = cv2.imread("../images/family.jpg")
        result = mtcnn.detect_faces(testpic)  # type: list

        first = result[0]

        self.assertIn('box', first)
        self.assertIn('keypoints', first)
        self.assertTrue(len(first['box']), 1)
        self.assertTrue(len(first['keypoints']), 5)

        keypoints = first['keypoints']  # type: dict
        self.assertIn('nose', keypoints)
        self.assertIn('mouth_left', keypoints)
        self.assertIn('mouth_right', keypoints)
        self.assertIn('left_eye', keypoints)
        self.assertIn('right_eye', keypoints)

        self.assertEqual(len(keypoints['nose']), 2)
        self.assertEqual(len(keypoints['mouth_left']), 2)
        self.assertEqual(len(keypoints['mouth_right']), 2)
        self.assertEqual(len(keypoints['left_eye']), 2)
        self.assertEqual(len(keypoints['right_eye']), 2)

    def test_mtcnn_multiple_instances1(self):
        """
        Multiple instances of MTCNN can be created in the same thread.
        """
        detector_1 = MTCNN(steps_threshold=[.2, .5, .5])
        testpic = cv2.imread("../images/family.jpg")

        faces_1 = detector_1.detect_faces(testpic)
        print(f"detector_1 detected faces: {len(faces_1)}") 
 
        self.visualize_faces(testpic, faces_1, "Detector 1") 

    def test_mtcnn_multiple_instances2(self):
 
        detector_2 = MTCNN(steps_threshold=[.1, .1, .2])
        testpic = cv2.imread("../images/family.jpg")
        faces_2 = detector_2.detect_faces(testpic)
 
        print(f"detector_2 detected faces: {len(faces_2)}")
 
        self.visualize_faces(testpic, faces_2, "Detector 2")

    def visualize_faces(self, image, faces, title):
        for face in faces:
            box = face['box']
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

    @classmethod
    def tearDownClass(cls):
        global mtcnn
        del mtcnn

if __name__ == '__main__':
    unittest.main()
