FROM python:3.11

# Set shell to bash
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]

# Update system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y pip git 

# Install difi
RUN mkdir -p /code/
WORKDIR /code/
ADD . /code/
RUN pip install -e .[dev]
