#! /bin/bash

mkdir logs plugins models
echo -e "AIRFLOW_UID=$(id -u)" > .env
echo -e "AIRFLOW_GID=0" >> .env

docker-compose up airflow-init
docker-compose build
docker-compose up