FROM tensorflow/tensorflow:1.8.0
WORKDIR /app
ADD . /app
RUN pip install pyyaml paho-mqtt
CMD ["python", "image-inference.py"]
