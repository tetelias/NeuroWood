FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 wget unzip

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir albumentations nose timm wandb
RUN conda install -y pandas
