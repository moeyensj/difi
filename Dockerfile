FROM ubuntu:latest

# Update system dependencies
RUN apt-get update \
	&& apt-get upgrade -y

# Install system dependencies
RUN apt-get install -y git python3 python3-pip python3-dev

# Upgrade pip to latest version
RUN pip install --upgrade pip

RUN mkdir /code/
ADD . /code/
WORKDIR /code/
RUN pip install -e .[tests]

# Install pre-commit hooks
RUN pre-commit install
