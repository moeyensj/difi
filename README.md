# difi
Did I Find It?  

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://img.shields.io/badge/Python-3.8%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/152989392.svg)](https://zenodo.org/badge/latestdoi/152989392)  
[![docker - Build, Lint, and Test](https://github.com/moeyensj/difi/actions/workflows/docker-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/difi/actions/workflows/docker-build-lint-test.yml)
[![conda - Build, Lint, and Test](https://github.com/moeyensj/difi/actions/workflows/conda-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/difi/actions/workflows/conda-build-lint-test.yml)
[![pip - Build, Lint, Test, and Coverage](https://github.com/moeyensj/difi/actions/workflows/pip-build-lint-test-coverage.yml/badge.svg)](https://github.com/moeyensj/difi/actions/workflows/pip-build-lint-test-coverage.yml)  
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/difi/badge.svg?branch=main)](https://coveralls.io/github/moeyensj/difi?branch=main)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/difi)](https://hub.docker.com/r/moeyensj/difi)  
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/difi/badges/version.svg)](https://anaconda.org/moeyensj/difi)
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/difi/badges/platforms.svg)](https://anaconda.org/moeyensj/difi)
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/difi/badges/downloads.svg)](https://anaconda.org/moeyensj/difi)

## About
`difi` is a simple package that takes pre-formatted linkage information from software such as [MOPS](https://github.com/lsst/mops_daymops), [pytrax](https://github.com/pytrax/pytrax), or [THOR](https://github.com/moeyensj/thor) and analyzes which objects have been found given a set of known labels (or truths). A key performance criteria is that `difi` needs to be fast by avoiding Python for loops and instead uses clever `pandas.DataFrame` manipulation.

## Installation

### Released Versions

#### Anaconda
`difi` can be downloaded directly from anaconda:  
```conda install -c moeyensj difi```

Or, if preferred, installed into its own environment via:  
```conda create -n difi_py310 -c moeyensj difi python=3.10```

#### Pip
`difi` is also available from the Python package index:  
```pip install difi```

#### Docker

A Docker container with the latest version of the code can be pulled using:  
```docker pull moeyensj/difi:latest```

To run the container:  
```docker run -it moeyensj/difi:latest```

The difi code is installed the /projects directory, and is by default also installed in the container's Python installation.

### Latest From Source

#### Anaconda
Clone this repository using either `ssh` or `https`. Once cloned and downloaded, `cd` into the repository.

To install difi in its own `conda` environment please do the following:  
```conda create -n difi_py310 -c defaults -c conda-forge --file requirements.txt python=3.10```  

Or, to install difi in a pre-existing `conda` environment called `difi_py310`:  
```conda activate difi_py310```  
```conda install -c defaults -c conda-forge --file requirements.txt```  

#### Pip

Or, to install `difi` software using `pip`:  
```pip install .```

Or, if you would like to make an editable install then:  
```pip install -e .[tests]```

You should now be able to start Python and import difi.

#### Docker Compose

After cloning this repository, you can build a docker image that will allow you to develop the source code:

```docker compose build difi```

To run the docker container interatively with a terminal:

```docker compose run -it difi```

### Developing

If you would like to contribute to `difi`, please make sure to initialize `pre-commit`. Pre-commit will automatically lint and format
the source code after any changes have been staged for a commit. To load the appropriate hooks please run:

```pre-commit install```

## Example and Tutorial

The example below can be found in greater detail in this [Jupyter Notebook](https://github.com/moeyensj/difi/tree/main/examples/tutorial.ipynb).
