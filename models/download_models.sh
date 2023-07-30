#!/usr/bin/env sh


mkdir -p ./models
for url in \
    https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth \
    https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth; do
    wget \
        -nc \
        -c \
        --directory-prefix ./models \
        "${url}"

done
