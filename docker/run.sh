#!/usr/bin/env sh


docker \
    run \
        --shm-size='16gb' \
        --gpus 0 \
        --device /dev/nvidia0 \
        --device /dev/nvidia-uvm \
        --device /dev/nvidia-uvm-tools \
        --device /dev/nvidiactl \
        -it \
        --rm \
        --gpus all \
        --net host \
        -v $(pwd)/data:/data \
        -v $(pwd)/models:/models \
        -v $(pwd)/src:/src \
        mvo
