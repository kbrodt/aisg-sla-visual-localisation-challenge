import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import functools
import random
import sys
from pathlib import Path

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

from DeDoDe import dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from roma import roma_outdoor

import normalised_rpe
from reader import DatasetReader
from micro_bundle_adjustment import (
    bundle_adjust,
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)


def plot_traj(trackedPoints, groundtruthPoints, title="Trajectory", save_path="traj.png", x=0):
    fig = plt.figure(figsize=(12, 8))

    spec = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(spec[0, 0], aspect="equal")

    y = 1
    ax.scatter(trackedPoints[[0],  x], trackedPoints[[0],  y], c="blue")
    ax.plot(trackedPoints[:, x], trackedPoints[:, y], c='blue', label="Tracking")

    x = 0
    ax.scatter(groundtruthPoints[[0],  x], groundtruthPoints[[0],  y], c="green")
    ax.plot(groundtruthPoints[:, x], groundtruthPoints[:, y], c='green', label="Ground truth")

    ax.set_title(title)
    ax.legend()

    plt.savefig(save_path)
    #plt.show()
    #fig.show()
    plt.close(fig)


def set_seed(seed=314159):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)

    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def create_groups(df, sec_thresh=30):
    df["Timestamp_diff"] = df["Timestamp"].diff().dt.total_seconds().fillna(value=0)
    df["IsNewFrame"] = df["Timestamp_diff"] >= sec_thresh

    groups = []
    g = 0
    for item in df.itertuples():
        if item.IsNewFrame:
            g += 1

        groups.append(g)

    df["Traj"] = groups

    return df


def projection(X, r, t, K):
    if len(X.shape) == 1:
        X = X[None]

    N, D = X.shape
    if len(r.shape) == 1:
        r = r.expand(N, D)
        t = t.expand(N, D)

    R = angle_axis_to_rotation_matrix(r)
    X = X.unsqueeze(-1)
    t = t.unsqueeze(-1)
    KRT = K @ R.transpose(-1, -2)
    x = KRT @ (X - t)
    x = x.squeeze(-1)

    return x[...,:2]/x[...,[2]]


def gold_standard_residuals(X, r, t, x_a, x_b, K):
    r_a = x_a - projection(X, torch.zeros_like(r), torch.zeros_like(t), K)
    r_b = x_b - projection(X, r, t, K)

    return torch.cat((r_a, r_b), dim=1)


def projection_matrix(R, T, K):
    P = np.zeros((3, 4))
    P[:3, :3] = R.T
    P[:3, 3] = -R.T @ T.ravel()
    P = K @ P

    return P


def bundle_adj(R, t, _K, q_curr, q_prev, device="cuda", dtype=torch.float32):
    curr_P = projection_matrix(R, t, _K)
    K = np.pad(_K, ((0, 0), (0, 1)))

    X = cv2.triangulatePoints(K, curr_P, q_prev.T, q_curr.T)
    X /= X[3]
    X = X[:3]
    X = X.T

    noisy_scene_points = torch.from_numpy(X).to(device=device, dtype=dtype)

    image_A_points = torch.from_numpy(q_prev).to(device=device, dtype=dtype)
    image_B_points = torch.from_numpy(q_curr).to(device=device, dtype=dtype)

    Rt = torch.from_numpy(R).to(device=device, dtype=dtype)
    noisy_r = rotation_matrix_to_angle_axis(Rt.unsqueeze(0)).squeeze(0)
    noisy_t = torch.from_numpy(t.ravel()).to(device=device, dtype=dtype)

    K = torch.from_numpy(_K).to(device=device, dtype=dtype)

    _, r_hat, t_hat = bundle_adjust(
        functools.partial(gold_standard_residuals, K=K),
        noisy_scene_points,
        noisy_r,
        noisy_t,
        image_A_points,
        image_B_points,
    )

    R = angle_axis_to_rotation_matrix(r_hat.unsqueeze(0)).squeeze(0)
    R = R.cpu().numpy()
    t = t_hat.unsqueeze(1).cpu().numpy()

    return R, t


def process(dataset_reader, roma_model, descriptor, matcher, device):
    K = dataset_reader.readCameraMatrix()

    prev_frame_BGR = dataset_reader.readFrame(0)
    prev_im = Image.fromarray(
        cv2.rotate(
            cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2RGB),
            cv2.ROTATE_90_CLOCKWISE,
        ),
    )

    gt_poses = []
    gt_pos, _ = dataset_reader.readGroundtuthPosition(0)
    gt_poses.append(gt_pos)

    gt_rot = dataset_reader.readGTangle(0)
    gt_rot = Rotation.from_euler("xyz", gt_rot, degrees=True)

    estimated_poses = [np.array(gt_pos)]
    estimated_rots = [gt_rot.as_matrix()]

    # Process next frames
    for frame_no in tqdm.trange(1, len(dataset_reader)):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)

        # Feature detection & filtering
        curr_im = Image.fromarray(
            cv2.rotate(
                cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2RGB),
                cv2.ROTATE_90_CLOCKWISE,
            ),
        )
        W_A, H_A = curr_im.size
        W_B, H_B = prev_im.size

        warp, certainty = roma_model.match(curr_im, prev_im, device=device)
        matches, certainty = roma_model.sample(warp, certainty)

        curr_kps = matches[..., :2].unsqueeze(0)
        P_A = certainty.unsqueeze(0)
        prev_kps = matches[..., 2:].unsqueeze(0)
        P_B = certainty.unsqueeze(0)

        curr_descr = descriptor.describe_keypoints_from_path(
            curr_im,
            curr_kps,
            H=768,
            W=768,
        )["descriptions"]
        prev_descr = descriptor.describe_keypoints_from_path(
            prev_im,
            prev_kps,
            H=768,
            W=768,
        )["descriptions"]

        curr_matches, prev_matches, _ = matcher.match(
            curr_kps, curr_descr,
            prev_kps, prev_descr,
            P_A=P_A, P_B=P_B,
            normalize=True,
            inv_temp=20,
            threshold=0.1,  #Increasing threshold -> fewer matches, fewer outliers
        )

        curr_matches, prev_matches = matcher.to_pixel_coords(curr_matches, prev_matches, H_A, W_A, H_B, W_B)
        curr_points, prev_points = curr_matches.cpu().numpy(), prev_matches.cpu().numpy()

        # rotate kps coordinates by -90 degrees
        a = curr_points.copy()
        curr_points[:, 0] = a[:, 1]
        curr_points[:, 1] = W_A - a[:, 0]

        a = prev_points.copy()
        prev_points[:, 0] = a[:, 1]
        prev_points[:, 1] = W_B - a[:, 0]

        if len(curr_points) < 8:
            estimated_poses.append(np.zeros(3))
            estimated_rots.append(np.eye(3))

            prev_im = curr_im

            continue

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)

        mask = mask.ravel() == 1
        prev_points = prev_points[mask]
        curr_points = curr_points[mask]

        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)

        # local bundle adjustment
        R, T = bundle_adj(R, T, K, curr_points, prev_points, device=device)

        # Read groundtruth translation T and absolute scale for computing trajectory
        gt_pos, scale = dataset_reader.readGroundtuthPosition(frame_no)
        gt_poses.append(gt_pos)

        estimated_poses.append(estimated_poses[-1] + scale * estimated_rots[-1] @ T.ravel())
        estimated_rots.append(estimated_rots[-1].dot(R))

        prev_im = curr_im

    gt_poses = np.array(gt_poses)
    estimated_poses = np.array(estimated_poses)
    estimated_rots = np.array(estimated_rots)

    return estimated_poses, estimated_rots, gt_poses


def main():
    set_seed(seed=0)

    data_dir = Path("./data/")
    models_dir = Path("./models")

    tid = None
    if tid is not None:
        imgs_dir = data_dir / f"train_images-{tid}"
        df_path = data_dir / "train_labels.csv"
    else:
        imgs_dir = data_dir / "test_images/"
        df_path = data_dir / "submission_format.csv"

    df = pd.read_csv(df_path)
    usecols = list(df.columns)[:-6]

    if tid is not None:
        df = df[df["TrajectoryId"] == tid].copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.sort_values("Timestamp", inplace=True)
    df = create_groups(df)
    gdf = df.groupby("Traj")

    #if tid is None:
    gdf = [(0, df)]

    device = torch.device("cuda")
    roma_model = roma_outdoor(
        weights=torch.load(models_dir / "roma_outdoor.pth", map_location=device),
        dinov2_weights=torch.load(models_dir / "dinov2_vitl14_pretrain.pth", map_location=device),
        device=device,
    )
    descriptor = dedode_descriptor_B(
        weights=torch.load(models_dir / "dedode_descriptor_B.pth", map_location=device),
    )
    matcher = DualSoftMaxMatcher()

    if len(sys.argv) > 1:
        gi = int(sys.argv[1])
    else:
        gi = None

    all_e_rots = []
    all_e_pos = []
    dfs = []
    for i, sdf in gdf:
        if gi is not None and i != gi:
            continue

        dataset_reader = DatasetReader(imgs_dir, data_dir / "intrinsic_parameters.json", sdf)
        estimated_poses, estimated_rots, gt_poses = process(
            dataset_reader,
            roma_model,
            descriptor,
            matcher,
            device,
        )

        all_e_rots.append(estimated_rots)
        all_e_pos.append(estimated_poses)
        dfs.append(sdf)

        r = Rotation.from_matrix(estimated_rots)
        gt_y = np.stack(
            [
                sdf.Easting,
                sdf.Northing,
                sdf.Height,
                sdf.Roll,
                sdf.Pitch,
                sdf.Yaw,
            ],
            axis=1,
        )
        theta = r.as_euler("xyz", degrees=True)
        pred_y = np.hstack([estimated_poses, theta])
        e0 = normalised_rpe.normalised_relative_pose_errors(gt_y, np.zeros_like(pred_y))
        e = normalised_rpe.normalised_relative_pose_errors(gt_y, pred_y)
        print(e0)
        print(e)
        print()

        plot_traj(
            estimated_poses,
            gt_poses,
            title=f"RE: {e['rotation_error']:.5f} (SC {e0['rotation_error']:.5f})",
            save_path=f"traj{tid}_{sdf.Traj.iloc[0]:0>3}.png",
            x=2 if tid is None else 0,
        )

    df = pd.concat(dfs, axis=0)
    all_e_rots = np.concatenate(all_e_rots, axis=0)
    all_e_pos = np.concatenate(all_e_pos, axis=0)
    r = Rotation.from_matrix(all_e_rots)
    gt_y = np.stack(
        [
            df.Easting,
            df.Northing,
            df.Height,
            df.Roll,
            df.Pitch,
            df.Yaw,
        ],
        axis=1,
    )

    theta = r.as_euler("xyz", degrees=True)
    pred_y = np.hstack([all_e_pos, theta])
    target_cols = [
        "Easting",
        "Northing",
        "Height",
        "Roll",
        "Pitch",
        "Yaw",
    ]
    df[target_cols] = pred_y

    submission = pd.read_csv(
        df_path,
        usecols=usecols,
    )
    submission = pd.merge(
        df[usecols + target_cols],
        submission,
        on="Filename",
        how="left",
        suffixes=("", "_y"),
    )[usecols + target_cols].copy()

    pred_y = np.stack(
        [
            submission.Easting,
            submission.Northing,
            submission.Height,
            submission.Roll,
            submission.Pitch,
            submission.Yaw,
        ],
        axis=1,
    )
    if tid is None:
        submission.to_csv("./submission_roma_outdoor_1344_dedode_B_768_portraid_1_1344_ba.csv", index=False)
    else:
        submission.to_csv(f"./traj{tid}.csv", index=False)

    e0 = normalised_rpe.normalised_relative_pose_errors(gt_y, np.zeros_like(pred_y))
    e = normalised_rpe.normalised_relative_pose_errors(gt_y, pred_y)
    print(e0)
    print(e)


if __name__ == "__main__":
    main()
