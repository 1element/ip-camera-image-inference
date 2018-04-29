# ==============================================================================
# IP camera image inference.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""IP camera image inference.

This Python script is supposed to be run as a systemd service.

It will subscribe to an MQTT broker to receive new JPEG images and run 
TensorFlow with a retrained inception model to classify the provided image 
into 'nothing' or 'person' along with their probability score.

Depending on the score and configured threshold the image will either be
discarded or published to a specified MQTT topic.

Please see the provided README.md for a detailed description of how to use
this script to perform image recognition.
"""

from __future__ import absolute_import

import yaml
import logging
import os
import datetime

import Queue

import numpy as np
import tensorflow as tf
import paho.mqtt.client as paho

NUM_PREDICTIONS = 2
config = None
labels = None
mqtt_client = None

image_queue = Queue.Queue()


def load_config():
  """Load config yaml file."""
  global config
  with open('config.yml', 'r') as file:
    config = yaml.load(file)


def configure_logging():
  """Configure logging."""
  numeric_level = getattr(logging, config['logging']['level'])
  logging.basicConfig(level=numeric_level,
    filename=config['logging']['filename'],
    format='%(asctime)s %(levelname)s: %(message)s')


def create_graph():
  """Creates a tensorflow graph from saved GraphDef file."""
  with tf.gfile.FastGFile(os.path.join(
      config['inference']['model_dir'], 'output_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def load_labels():
  """Read in labels, one label per line."""
  filename = os.path.join(config['inference']['model_dir'], 'output_labels.txt')
  global labels
  labels = [line.rstrip() for line in tf.gfile.FastGFile(filename)]


def save_image(image):
  """Save image file if enabled in configuration."""
  if config['save_images']['enabled']:
    directory = config['save_images']['destination']
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.jpg'
    destination = os.path.join(directory, filename)
    logging.debug('saving image to %s', destination)
    f = open(destination, 'wb')
    f.write(image)
    f.close


def mqtt_connect():
  """Create MQTT client and connect to broker."""
  global mqtt_client
  logging.debug('connecting to mqtt broker %s', config['mqtt']['host'])
  mqtt_client = paho.Client()
  mqtt_client.on_connect = mqtt_on_connect
  mqtt_client.on_message = mqtt_on_message
  mqtt_client.username_pw_set(config['mqtt']['username'], config['mqtt']['password'])
  mqtt_client.connect(config['mqtt']['host'], config['mqtt']['port'])
  mqtt_client.loop_start()


def mqtt_on_connect(client, userdata, flags, rc):
  """Callback on MQTT connection."""
  logging.debug('successfully connected to mqtt broker')
  client.subscribe(config['mqtt']['subscribe_topic'])


def mqtt_on_message(client, userdata, msg):
  """Callback on MQTT message."""
  logging.debug('mqtt message received for topic %s', msg.topic)
  image_queue.put(msg.payload)


def mqtt_publish(image):
  """Publish image to MQTT broker."""
  logging.debug('publishing image to mqtt broker topic %s', 
    config['mqtt']['publish_topic'])
  mqtt_client.publish(config['mqtt']['publish_topic'], image)


def serve_inference_requests():
  """Infinite loop serving inference requests."""
  global image_queue

  with tf.Session() as sess:
    while True:
      image_data = image_queue.get()

      tensor = sess.graph.get_tensor_by_name('final_result:0')
      predictions = sess.run(tensor, {'DecodeJpeg/contents:0': image_data})
      predictions = np.squeeze(predictions)

      top_k = predictions.argsort()[-NUM_PREDICTIONS:][::-1]

      human_string = labels[top_k[0]]
      score = predictions[top_k[0]]
      logging.info('%s classified with score %.5f', human_string, score)

      emit_image = False
      if human_string != 'nothing':
        emit_image = True
        logging.debug('emitting image cause %s was detected', human_string)
      elif score <= config['inference']['threshold']:
        emit_image = True
        logging.debug('emitting image cause score %.5f is below threshold of %s',
          score, config['inference']['threshold'])
      else:
        logging.debug('image not emitted, cause nothing was detected with a probability of %.5f',
          score)

      if emit_image:
        mqtt_publish(image_data)
      else:
        save_image(image_data)


def main(_):
  # disable tensorflow compilation warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  create_graph()
  load_labels()

  serve_inference_requests()


if __name__ == '__main__':
  load_config()
  configure_logging()

  mqtt_connect()

  # run tensorflow main app
  tf.app.run(main=main)
