# IP Camera Image Inference

Python script using machine learning to classify images from IP cameras.


## Introduction

This Python script is meant to be executed as a systemd service. It will 
subscribe an MQTT broker to receive JPEG images from an IP camera.

The machine learning framework TensorFlow is used to classify images.
A deep convolutional neural network (CNN) based on the Inception v3 model 
needs to be retrained with a set of images from your IP camera to classify 
the image either as *person* or as *nothing*.

If a person is detected in the image, the Python script will publish the 
image to a specified MQTT topic. A threshold can be configured for this 
to publish even images with a lower probability score.

This script was designed to work on a Raspberry Pi 3 (ARM architecture), 
but it also runs on x86_64 and should actually run on all platforms where 
TensorFlow and Python are available.

On a Raspberry Pi 3 this is pretty slow (roughly three seconds per image). 
So make sure you have a good pre-processing of your IP camera feed and 
deliver only images where potentially movement was detected. You could 
either use the built-in motion detection of your IP camera or preferable 
a third party software detection like [Motion](https://motion-project.github.io/) for this.


## Dependencies

* TensorFlow
* Python 2.7
* Python PyYAML (pyyaml) and Eclipse Paho MQTT (paho-mqtt) packages

All dependencies can be installed automatically by executing `./install.sh`.


## Usage

### Installation

Clone this repository:

```
git clone https://github.com/1element/ip-camera-image-inference.git
```

Install dependencies:

```
./install.sh
```


### Train your model

The TensorFlow Inception v3 model is used to classify images either as
*person* or as *nothing*. For accurate scoring you need to retrain the 
final layer of the Inception model.

So before you can start `retrain.py` you need a set of images from your 
IP camera that represent the two classes *person* and *nothing*. 
Copy these images to the `retrain/person/` and `retrain/nothing/` 
directories.

After this execute:

```
python retrain.py --image_dir=/path/to/ip-camera-image-inference/retrain
```

This will take some time, especially if you run it on a Raspberry Pi. 
Consider running the training on a faster machine if you can.

Copy the output graph and labels from the temporary directory of 
your system to the model directory:

```
cp /tmp/output_graph.pb /path/to/ip-camera-image-inference/model/
cp /tmp/output_labels.txt /path/to/ip-camera-image-inference/model/
```

If you are interested in details of the retraining process refer 
to the official [TensorFlow image retraining tutorial](https://www.tensorflow.org/tutorials/image_retraining).


### Configuration

Edit the provided configuration file `config.yml`. The TensorFlow model 
directory and your MQTT broker connection as well as a few other settings 
can be configured there.


### Run as service

The Python script `image-inference.py` should be executed as a systemd 
service. There is a template file `image-inference.service` for this.
Change the paths in this file according to your environment.

Afterwards use the following commands to set up the service:

```
# copy service configuration
sudo cp image-inference.service /etc/systemd/system/

# reload systemd
sudo systemctl daemon-reload

# enable auto start
sudo systemctl enable image-inference.service

# start service now
sudo systemctl start image-inference.service

# show status
sudo systemctl status image-inference.service
```


## Previous versions

There is a previous version using a directory observer to receive images 
and FTP to distribute the classified image. If you are interested in this 
checkout the tag `v1.0.0`.


## Contributions

This script was written for my own purpose, therefore it lacks in 
features and configuration abilities. In case it might be useful for 
someone, I decided to share it. However, pull requests are welcome.


## License

This project is licensed under the terms of the [Apache License, Version 2.0](https://github.com/1element/ip-camera-image-inference/blob/master/LICENSE).
