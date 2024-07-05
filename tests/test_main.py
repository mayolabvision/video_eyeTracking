import os
import subprocess

def test_main():
    test_video_path = os.path.join(os.path.dirname(__file__), 'test_video.avi')
    command = [
        "python", "-m", "face_gazeTracking.main",
        "--direct_video_path", test_video_path,
        "test_video.mp4"
    ]
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0
    print(result.stdout.decode())
    print(result.stderr.decode())

if __name__ == "__main__":
    test_main()
