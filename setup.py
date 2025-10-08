from setuptools import setup, find_packages

setup(
    name="oureyes",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "oureyes-main = oureyes.main:main",
        ]
    },
)
