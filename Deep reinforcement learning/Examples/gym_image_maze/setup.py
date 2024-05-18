from setuptools import setup

try:
    with open('requirements.txt', 'r') as file:
        requirements = file.read()
except FileNotFoundError:
    requirements = []

setup(name='gym_image_maze',
      version='0.0.1',
      install_requires=requirements)