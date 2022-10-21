FROM sailugr/sinergym:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.9 -y

COPY requirements/requirements_dev.txt /tmp/pip-tmp/
RUN pip3 install --upgrade pip
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements_dev.txt \
    && rm -rf /tmp/pip-tmp
