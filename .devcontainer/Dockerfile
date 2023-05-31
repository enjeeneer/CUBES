FROM sailugr/sinergym:latest

# add python 3.9 to image
# note: noninteractive mode allows for python installation
# (otherwise a question would block process)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install python3.9 -y \
    && apt install python3.9-distutils -y \
    && apt-get install -y python3.9-tk \
    && python3.9 -m pip install --upgrade pip \
    && python3.9 -m pip install -e /sinergym

# install 3.9 requirements
COPY requirements/requirements_dev.txt /tmp/pip-tmp/
RUN python3.9 -m pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_dev.txt \
    && rm -rf /tmp/pip-tmp


# old default python version (3.10) installation of requirements
# TODO: remove in future when default python version fully changed
COPY requirements/requirements_dev.txt /tmp/pip-tmp/
RUN pip3 install --upgrade pip
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_dev.txt \
    && rm -rf /tmp/pip-tmp

#install cubes package
#RUN pip install -e .
#RUN python3.9 -m pip install -e .