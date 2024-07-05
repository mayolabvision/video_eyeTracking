import unittest
from your_project.preprocessing import step2

class TestStep2(unittest.TestCase):
    def test_preprocess(self):
        video_path = "path/to/test/video.mp4"
        result = step1.preprocess(video_path)
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()