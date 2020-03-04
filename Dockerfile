# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
# Copyright (c) 2020, Lee Butterman. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter
RUN apt-get update && \
    apt-get install -y cmake && \
    pip install dlib && \
    apt-get remove -y cmake && \
    rm -rf /var/lib/apt/lists/*
RUN pip install keras pillow tqdm requests imutils opencv-python-headless
RUN pip install flask flask-cors gunicorn
WORKDIR /app
COPY . .
ENV FLASK_ENV=production
RUN mv ./dot-keras ~/.keras
CMD source /etc/bash.bashrc && \
    jupyter notebook --notebook-dir=. --ip 127.0.0.1 --no-browser --allow-root
