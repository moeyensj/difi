from setuptools import setup

setup(
   name="difi",
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
   use_scm_version=True,
   setup_requires=["pytest-runner", "setuptools_scm"],
   tests_require=["pytest"],
)
