#!/bin/bash

LOG="./install.log"

install_system_dependencies() {
  sudo apt-get update
  # python 2.7
  sudo apt-get install -y python-dev python-pip
}

install_python_dependencies() {
  sudo -H pip install pyyaml paho-mqtt
}

install_tensorflow() {
  TENSORFLOW_VERSION="1.0.1"
  TENSORFLOW_PKG_ARM="https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v${TENSORFLOW_VERSION}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_armv7l.whl"
  TENSORFLOW_PKG_X86="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl"
  ARCH=`uname -m`
  if [[ $ARCH == arm* ]]; then
    sudo -H pip install --upgrade $TENSORFLOW_PKG_ARM
  elif [[ $ARCH == x86* ]]; then
    sudo -H pip install --upgrade $TENSORFLOW_PKG_X86
  else
    echo "Unknown architecture $ARCH. Please install tensorflow manually."
    exit 1
  fi
}

install_system_dependencies 2>&1 | tee -a $LOG
install_python_dependencies 2>&1 | tee -a $LOG
install_tensorflow 2>&1 | tee -a $LOG

echo "Installation is completed." | tee -a $LOG
