#!/usr/bin/env sh


git clone https://github.com/Parskatt/RoMa.git
git -C RoMa checkout 7f8b2e455984c82d957e3318b963cf6673ddaba4
patch -p0 -d "RoMa" < ../patches/roma.patch
pip install -v -e ./RoMa

git clone https://github.com/Parskatt/DeDoDe.git
git -C DeDoDe checkout d49fe2f8a7a15f8959107ee7a851ced6f54e7621
patch -p0 -d "DeDoDe" < ../patches/dedode.patch
pip install -v -e ./DeDoDe

git clone https://github.com/Parskatt/micro-bundle-adjustment.git
git -C micro-bundle-adjustment checkout 934eff87efaf4af3f2f79cfba2de7e60ae4ea5a8
patch -p0 -d "micro-bundle-adjustment" < ../patches/micro_bundle_adjustment.patch
pip install -v -e ./micro-bundle-adjustment
