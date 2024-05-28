import unittest
from unittest.mock import patch
from io import StringIO
import sys
import os
import warnings

# Adjust the system path to include the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the analyze_gender function from the gender_recog module
from gender_recog import analyze_gender

class TestGenderRecognition(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_analyze_gender(self, mock_stdout):
        # Suppress the ResourceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
        
            # Path to the directory containing the images
            image_folder = os.path.join(os.path.dirname(__file__), '../images')
            image_names = os.listdir(image_folder)
            
            # Iterate over each image and analyze it
            for image_name in image_names:
                image_path = os.path.join(image_folder, image_name)
                print(f"Analyzing {image_name}...")
                analyze_gender(image_path)

        # Get the printed output
        output = mock_stdout.getvalue()

        # Check for specific expected results in the output
        self.assertIn("Detected", output)  # Check that faces were detected in at least one image
        self.assertIn("Gender:", output)   # Check that gender was analyzed
        self.assertIn("No face detected", output)  # Check that no face detection was handled

if __name__ == "__main__":
    unittest.main()
