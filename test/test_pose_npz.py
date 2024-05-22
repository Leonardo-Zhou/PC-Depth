import os
import numpy as np
import sys

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        # cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        # cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def eval(pred_path, gt_path, dataset):

    pred_poses = np.load(pred_path, fix_imports=True, encoding='latin1')["data"]
    gt_local_poses = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print(pred_poses.shape, gt_local_poses.shape)

    ates = []
    res = []
    num_frames = gt_local_poses.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        res.append(compute_re(local_rs, gt_rs))

    # print("Trajectory error: {:0.4f}, std: {:0.4f}".format(np.mean(ates), np.std(ates)))
    # print("Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res)*100, np.std(res)))
    print("& {:0.4f} & {:0.4f} \\\\".format(np.mean(ates), np.mean(res)*100))


if __name__ == "__main__":
    pred_dir = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == "c3vd":
        files = [
            "cecum_t2_a.npz",
            # "desc_t4_a_up.npz",
            "sigmoid_t3_a.npz",
            "trans_t2_a.npz",
        ]
        gt_dir = "test/c3vd_pose5_gt"
        for file in files:
            print(file)
            eval(os.path.join(pred_dir, file), os.path.join(gt_dir, file), dataset)
    elif dataset == "SCARED":
        gt_path = "test/SCARED_pose_gt/gt_depths.npz"
        eval(pred_dir, gt_path, dataset)
