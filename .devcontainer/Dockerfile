FROM sailugr/sinergym:v2.2.0

# add python3.9 3.9 to image
# note: noninteractive mode allows for python3.9 installation
# (otherwise a question would block process)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y \
    && apt install python3-distutils -y \
    && apt-get install -y python3-tk \
    && pip install --upgrade pip \
    && pip install -e /sinergym

# install 3.9 requirements
COPY requirements/requirements_dev.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_dev.txt \
    && rm -rf /tmp/pip-tmp


# old default python3.9 version (3.10) installation of requirements
# TODO: remove in future when default python3.9 version fully changed
#COPY requirements/requirements_dev.txt /tmp/pip-tmp/
#RUN pip3 install --upgrade pip
#RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_dev.txt \
#    && rm -rf /tmp/pip-tmp

#install cubes package
#RUN pip install -e .
#RUN python3.9 -m pip install -e .