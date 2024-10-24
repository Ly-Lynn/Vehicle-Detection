FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /co-detr
COPY . /co-detr/
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    apt-get update && \
    apt-get install -y \
        libatlas-base-dev \
        libblas3 \
        liblapack3 \
        liblapack-dev \
        libblas-dev \
        gfortran \
        libgl1-mesa-glx \
        tzdata \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        git && \
    ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata 
RUN pip install mmengine \
    mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    rm -rf mmdetection && \
    git clone https://github.com/open-mmlab/mmdetection.git &&\
    pip install -e . &&\
    pip install -r requirements.txt &&\
    pip install mmdet==2.25.3