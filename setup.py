from setuptools import setup


with open("README-frsystem.md", "r") as fh:
    long_description = fh.read()

setup(name='frsystem',
      version='1.1.2',
      description='A system for experimenting with face recognition',
      long_description = long_description,
      long_description_content_type = "text/markdown",
      url='https://github.com/amac-lfc/frsystem',
      author='Arnur Sabet',
      author_email='arnursabet@gmail.com',
      license='GNU GENERAL PUBLIC LICENSE',
      install_requires=["numpy","requests","mtcnn","opencv-python","scikit_learn","tensorflow"],
      packages=['frsystem'],
      zip_safe=False)

