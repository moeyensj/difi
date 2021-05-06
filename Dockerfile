FROM continuumio/miniconda3

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update apps
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda

# Download difi, create a Python 3.8 conda environment, install requirements, then install difi
RUN mkdir projects \
	&& cd projects \
	&& git clone https://github.com/moeyensj/difi.git --depth=1 \
	&& cd difi \
	&& conda install -c defaults -c conda-forge --file requirements.txt python=3.8 --y \
	&& python setup.py install
