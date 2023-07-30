#!/usr/bin/env sh


# create virutal env
python -m venv --clear venv

# activate venv
. venv/bin/activate

# update pip
pip install -U pip setuptools

# install deps
pip install -r requirements.txt

# RoMa installation may fail, so install RoMa manually
git clone https://github.com/Parskatt/RoMa.git /tmp/RoMa
git -C /tmp/RoMa/ checkout 7f8b2e455984c82d957e3318b963cf6673ddaba4
pip install -v -e /tmp/RoMa/

# patch libs
v=$(python -V | cut -d ' ' -f 2 | cut -d'.' -f 1-2)
patch -p0 -d "/tmp/RoMa" < ./patches/roma.patch
patch -p0 -d "venv/lib/python${v}/site-packages" < ./patches/dedode.patch
patch -p0 -d "venv/lib/python${v}/site-packages" < ./patches/micro_bundle_adjustment.patch

# perform inference (~1 hour)
python ./src/inference.py
