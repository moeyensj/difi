from setuptools import setup

setup(
   name="difi",
   version="1.0.0",
   license="BSD 3-Clause License",
   author="Joachim Moeyens",
   author_email="moeyensj@uw.edu",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   install_requires=[
       'numpy',
       'pandas',
       'pytest',
       'pytest-cov'
   ],
   url="https://github.com/moeyensj/difi",
   packages=["difi"],
   package_dir={"difi": "difi"},
   package_data={"difi": ["tests/*.txt"]},
   setup_requires=["pytest-runner"],
   tests_require=["pytest"],
)
