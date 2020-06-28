FROM python:3.7

RUN mkdir -p /hair_segmentation
WORKDIR /hair_segmentation
COPY . /hair_segmentation/
RUN python -m pip install -r /hair_segmentation/requirements.txt

VOLUME /data
VOLUME /output

