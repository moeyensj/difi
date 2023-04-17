FROM ubuntu:latest

# Update system dependencies
RUN apt-get update \
	&& apt-get upgrade -y

# Install system dependencies
RUN apt-get install -y git python3 python3-pip python3-dev

# Upgrade pip to the latest version and install pre-commit
RUN pip install --upgrade pip pre-commit

# Install pre-commit hooks (before difi is installed to cache this step)
# Remove the .git directory after pre-commit is installed as difi's .git
# will be added to the container
RUN mkdir /code/
COPY .pre-commit-config.yaml /code/
WORKDIR /code/
RUN git init . \
	&& pre-commit install-hooks \
	&& rm -rf .git

# Install difi
ADD . /code/
RUN pip install -e .[tests]
