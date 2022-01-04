From python:3.8

ENV PYTHONUNBUFFERED 1

RUN mkdir -p /graph_model_staging/ /graph_model_staging/logs/

WORKDIR /graph_model_staging/

ADD requirements.pip /graph_model_staging/

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.pip

RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_sm

ADD . /graph_model_staging/


