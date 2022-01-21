FROM ubuntu:20.04

RUN apt-get update && \
    apt install -y python3.8 python3-pip git

RUN mkdir /src
COPY requirements.txt /src/
RUN pip install -r /src/requirements.txt
COPY main_script.py /src/
WORKDIR /src

VOLUME /src/bank_scoring.csv
ENV AZURE_STORAGE_CONNECTION_STRING=

CMD python3 main_script.py