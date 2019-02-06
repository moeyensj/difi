# difi
Did I Find It?  
[![Build Status](https://travis-ci.com/moeyensj/difi.svg?branch=master)](https://travis-ci.com/moeyensj/difi)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/difi/badge.svg?branch=master)](https://coveralls.io/github/moeyensj/difi?branch=master)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## About
`difi` is a simple package that takes pre-formatted linkage information from software such as [MOPS](https://github.com/lsst/mops_daymops), [pytrax](https://github.com/pytrax/pytrax) or [thor](https://github.com/moeyensj/thor) and analyzes which objects have been found. A key performance criteria is that `difi` needs to be fast by avoiding Python for loops and instead uses clever `pandas.DataFrame` manipulation. 

## Installation
To install pre-requisite software using anaconda:

```conda install -c defaults -c conda-forge --file requirements.txt```


## Examples
See the examples subdirectory for specific examples on how to use `difi`. 
- MOPS: example_mops.ipynb
- pytrax: example_pytrax.ipynb
- thor: example_thor.ipynb


