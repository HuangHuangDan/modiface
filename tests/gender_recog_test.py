import unittest
import sys
import os

# Ensure the src folder is in the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gender_recog import analyze_and_modify_image

class TestGenderRecognition(unittest.TestCase):

    def test_analyze_and_modify_image(self):
        image_folder = "../images"
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            print(f"Testing {image_name}...")
            analyze_and_modify_image(image_path)

if __name__ == "__main__":
    unittest.main()
