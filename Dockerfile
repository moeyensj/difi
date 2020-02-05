FROM continuumio/miniconda3
MAINTAINER Joachim Moeyens <moeyensj@gmail.com>

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update apps
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda

# Download difi
RUN mkdir projects \
	&& cd projects \
	&& git clone https://github.com/moeyensj/difi.git --depth=1

# Create Python 3.6 conda environment and install requirements, then install difi
RUN cd projects/difi \
	&& conda install -c defaults -c conda-forge -c astropy --file requirements.txt python=3.6 --y \
	&& python setup.py install
