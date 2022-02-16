FROM jupyter/pyspark-notebook

RUN mkdir /home/jovyan/src/
COPY requirements.txt /home/jovyan/src/
RUN pip install -r ./src/requirements.txt
COPY etl.py app.py /home/jovyan/src/

VOLUME /home/jovyan/data

CMD python ./src/app.py