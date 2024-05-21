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
        MTCNN is able to detect faces and landmarks on an image
        :return:
        """
        ivan = cv2.imread("/Users/torryy/Documents/modiface/images/family.jpg")
        result = mtcnn.detect_faces(ivan)  # type: list
        self.assertEqual(len(result), 1)

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

    def test_detect_faces_invalid_content(self):
        """
        MTCNN detects invalid images
        :return:
        """
        ivan = cv2.imread("example.py")

        with self.assertRaises(InvalidImage):
            result = mtcnn.detect_faces(ivan)  # type: list

    def test_detect_no_faces_on_no_faces_content(self):
        """
        MTCNN successfully reports an empty list when no faces are detected.
        :return:
        """
        ivan = cv2.imread("no-faces.jpg")

        result = mtcnn.detect_faces(ivan)  # type: list
        self.assertEqual(len(result), 0)

    def test_mtcnn_multiple_instances(self):
        """
        Multiple instances of MTCNN can be created in the same thread.
        :return:
        """
        detector_1 = MTCNN(steps_threshold=[.2, .7, .7])
        detector_2 = MTCNN(steps_threshold=[.1, .1, .1])

        ivan = cv2.imread("/Users/torryy/Documents/modiface/images/family.jpg")

        faces_1 = detector_1.detect_faces(ivan)
        faces_2 = detector_2.detect_faces(ivan)

 
        print(f"detector_1 detected faces: {len(faces_1)}")
        print(f"detector_2 detected faces: {len(faces_2)}")
 
        self.visualize_faces(ivan, faces_1, "Detector 1")
        self.visualize_faces(ivan, faces_2, "Detector 2")

        self.assertEqual(len(faces_1), 1)
        self.assertGreater(len(faces_2), 1)

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
