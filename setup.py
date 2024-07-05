from setuptools import setup, find_packages

setup(
    name="your_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "your_project=your_project.main:main",
        ],
    },
)
