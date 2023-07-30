import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    with open(path) as fin:
        return json.load(fin)


class DatasetReader:
    def __init__(self, img_root, intrinsic_path, df):
        self.img_root = Path(img_root)

        self.intrinsic_params = read_json(intrinsic_path)

        self.df = df.copy()

        xyz = self.df[[
            "Easting",
            "Northing",
            "Height"
        ]].values

        self.df[[
            "Easting",
            "Northing",
            "Height"
        ]] = xyz - xyz[0]

    def __len__(self):
        return len(self.df)

    def readFrame(self, index=0):
        if index >= len(self.df):
            raise Exception("Cannot read frame number {}".format(index))

        path = self.img_root / self.df.Filename.iloc[index]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        ##img = img[:, :-259]  # {'rotation_error': 0.09078503679617365, 'translation_error': 7.54848666437204}
        #img = img[:, :-459]  # {'rotation_error': 0.056731466268408394, 'translation_error': 6.626816817046741}
        #img = img[:, :-359]  # {'rotation_error': 0.05478672505540393, 'translation_error': 6.5779195721161745}
        img = img[:, :-259]  # {'rotation_error': 0.05458509831928666, 'translation_error': 6.623291711211991}
        #img = img[:-259]  # {'rotation_error': 0.05458509831928666, 'translation_error': 6.623291711211991}
        #img = img[:, :-200]  #  {'rotation_error': 0.06159311640926715, 'translation_error': 7.119277518609054}
        #img = img[:, :-159]  # {'rotation_error': 0.06333369140054072, 'translation_error': 6.817316110802369}
        h, w = img.shape[:2]
        assert h == 2048
        #assert w == 2048
        #new_h = 512  # {'rotation_error': 0.06146115142843255, 'translation_error': 6.763517826956637}
        #new_h = 864  # {'rotation_error': 0.060445039032731594, 'translation_error': 7.144282569151933}
        #new_h = 1024  # {'rotation_error': 0.05458509831928666, 'translation_error': 6.623291711211991}
                      # {'rotation_error': 0.05090737288997111, 'translation_error': 6.200308690402812}
        #new_w = 1024
        #new_h = 1024 + 128  # {'rotation_error': 0.05952263258304835, 'translation_error': 6.618080674001098}
        #new_h = 1024 + 256  # {'rotation_error': 0.05838980461678012, 'translation_error': 6.9480397614865455}
                            # {'rotation_error': 0.04983314495355979, 'translation_error': 6.163375125815703}
        new_h = 1024 + 256 + 64  #
                                 # {'rotation_error': 0.0448765025026665, 'translation_error': 6.044135972551113}
                                 # dedode
                                 # {'rotation_error': 0.03739386947739949, 'translation_error': 5.9670244078575365}
        #new_h = 1024 + 512  # {'rotation_error': 0.0599296452696655, 'translation_error': 6.93007262543356}
                             # {'rotation_error': 0.05438638332855432, 'translation_error': 6.031760025272107}
        self.rh = new_h / h
        new_w = int(w * self.rh)
        self.rw = new_w / w
        new_shape = (new_w, new_h)
        img = cv2.resize(img, new_shape)
        h, w = img.shape[:2]
        assert w == new_w
        assert h == new_h

        return img

    def readCameraMatrix(self):
        self.readFrame(0)
        K = np.array(
            [
                [self.rw * self.intrinsic_params["fx"],                                      0, self.rw * self.intrinsic_params["Cx"]],
                [                                     0, self.rh * self.intrinsic_params["fy"], self.rh * self.intrinsic_params["Cy"]],
                [                                     0,                                     0,                                     1],
            ],
        )

        return K

    def readGroundtuthPosition(self, frameId):
        item = self.df.iloc[frameId]

        tx = item.Easting
        ty = item.Northing
        tz = item.Height
        position = (tx, ty, tz)

        if frameId > 0:
            item_prev = self.df.iloc[frameId - 1]
            tx_prev = item_prev.Easting
            ty_prev = item_prev.Northing
            tz_prev = item_prev.Height

            scale = math.sqrt((tx - tx_prev) ** 2 + (ty - ty_prev) ** 2  + (tz - tz_prev) ** 2)

        scale = 7.571103106907284

        return position, scale

    def readGTangle(self, frameId):
        item = self.df.iloc[frameId]
        x = item.Roll
        y = item.Pitch
        z = item.Yaw

        return x, y, z
