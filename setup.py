import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="py-sonaion-heatmaps",
    version="0.0.01",
    description="Library for creating heatmaps for Eyetracking or Visual Attention Data.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sonaion-eye-tracking/py-sonaion-heatmaps",
    author="Jonas Mucke",
    author_email="jonas.mucke@web.com",
    license="BSD 3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[""],
)