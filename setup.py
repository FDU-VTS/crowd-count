import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crowdcount", # Replace with your own username
    version="0.0.2.1",
    author="Fudan-VTS",
    author_email="sjchen18@fudan.edu.cn",
    description="package for crowd counting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FDU-VTS/crowd-count/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apple Public Source License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
