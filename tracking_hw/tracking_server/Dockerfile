FROM ubuntu:20.04

RUN apt-get update && \
    apt install -y sqlite python3.8 python3-pip

RUN mkdir /src
COPY tracking_server_requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

EXPOSE 5000

ENV AZURE_STORAGE_CONNECTION_STRING=
ENV AZURE_CONTAINER=

VOLUME /src/runs.db

CMD mlflow server --backend-store-uri sqlite:///runs.db --default-artifact-root $AZURE_CONTAINER --host 0.0.0.0 --port 5000