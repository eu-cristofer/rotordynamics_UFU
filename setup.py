from setuptools import setup, find_packages

# Package meta-data.
NAME = 'rotor_analysis'
DESCRIPTION = 'Tools for rotordynamic analysis'
EMAIL = 'cristofercosta@yahoo.com.br'
AUTHOR = 'Cristofer Antoni Souza Costa'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()

# What packages are optional?
EXTRAS = {
    "dev": [
        "jupyter-book",
        "opencv-python",
        "black",
    ]
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    url='https://eu-cristofer.github.io/rotordynamics_UFU/',
    license="Apache License 2.0",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

