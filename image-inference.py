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

It will observe a configured directory for new JPEG images and run 
TensorFlow with a retrained inception model to classify the provided image 
into 'nothing' or 'person' along with their probability score.

Depending on the score and configured threshold different actions will 
be taken, such as uploading the image to an FTP server.

Please see the provided README.md for a detailed description of how to use
this script to perform image recognition.
"""

from __future__ import absolute_import

import yaml
import logging
import os
import shutil

from ftplib import FTP
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

import Queue

import numpy as np
import tensorflow as tf

NUM_PREDICTIONS = 2
config = None
labels = None

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


def upload_image_ftp(image):
  """Upload image to FTP server."""
  logging.debug('uploading image %s to FTP server', image)
  ftp = FTP()
  ftp.connect(config['ftp']['host'], config['ftp']['port'])
  ftp.login(config['ftp']['username'], config['ftp']['password'])
  filename = os.path.basename(image)
  file = open(image, 'rb')
  ftp.storbinary('STOR ' + filename, file)
  file.close()
  ftp.quit()


def copy_image(image):
  """Copy image file if enabled in configuration."""
  if config['file_operations']['copy']:
    destination = config['file_operations']['copy_destination']
    logging.debug('copying image %s to %s', image, destination)
    shutil.copy(image, destination)


def delete_image(image):
  """Delete image file if enabled in configuration."""
  if config['file_operations']['delete']:
    logging.debug('deleting image %s', image)
    os.remove(image)


def serve_inference_requests():
  """Infinite loop serving inference requests."""
  global image_queue

  with tf.Session() as sess:
    while True:
      image = image_queue.get()
      image_data = tf.gfile.FastGFile(image, 'rb').read()

      tensor = sess.graph.get_tensor_by_name('final_result:0')
      predictions = sess.run(tensor, {'DecodeJpeg/contents:0': image_data})
      predictions = np.squeeze(predictions)

      top_k = predictions.argsort()[-NUM_PREDICTIONS:][::-1]

      human_string = labels[top_k[0]]
      score = predictions[top_k[0]]
      logging.info('%s classified with score %.5f for %s',
        human_string, score, image)

      emit_image = False
      if human_string != 'nothing':
        emit_image = True
        logging.debug('emitting image %s, cause %s was detected',
          image, human_string)
      elif score <= config['inference']['threshold']:
        emit_image = True
        logging.debug('emitting image %s, cause score %.5f is below threshold of %s',
          image, score, config['inference']['threshold'])
      else:
        logging.debug('image %s not emitted, cause nothing was detected with a probability of %.5f',
          image, score)

      if emit_image:
        upload_image_ftp(image)
        delete_image(image)
      else:
        copy_image(image)
        delete_image(image)


class EventHandler(PatternMatchingEventHandler):
  def on_created(self, event):
    """Event handler, invoked on creation of new images."""
    image_queue.put(event.src_path)
    logging.debug('put %s to image processing queue', event.src_path)


def main(_):
  # disable tensorflow compilation warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  create_graph()
  load_labels()

  serve_inference_requests()


if __name__ == '__main__':
  load_config()
  configure_logging()

  # observer handles events in a different thread
  observer = Observer()
  observer.schedule(EventHandler(['*.jpg']),
    path=config['inference']['image_watch_dir'], recursive=False)
  observer.start()

  # run tensorflow main app
  tf.app.run(main=main)
