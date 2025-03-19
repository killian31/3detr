import datetime
import logging
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.distributed.distributed_c10d import reduce

import utils.pc_util as pc_util
from utils.ap_calculator import APCalculator, flip_axis_to_depth, parse_predictions
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    barrier,
    is_primary,
    reduce_dict,
)
from utils.misc import SmoothedValue
from utils.pc_util import shift_scale_points


@torch.no_grad()
def visualize(
    args,
    model,
    dataset_config,
    dataset_loader,
):

    name_folder = args.checkpoint_dir
    name_folder = os.path.join(name_folder, args.visualize)

    if os.path.exists(name_folder):
        shutil.rmtree(name_folder)

    os.makedirs(name_folder)

    config_dict = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "use_cls_confidence_only": False,
        "conf_thresh": 0.25,
        "no_nms": False,
        "dataset_config": dataset_config,
    }

    net_device = next(model.parameters()).device

    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)

        if args.visualize is not None:
            predicted_boxes = outputs["outputs"]["box_corners"]
            sem_cls_probs = outputs["outputs"]["sem_cls_prob"]
            objectness_probs = outputs["outputs"]["objectness_prob"]
            point_cloud = batch_data_label["original_clouds"]
            point_cloud_modelo = batch_data_label["point_clouds"]
            point_cloud_dims_min = batch_data_label["point_cloud_dims_min"]
            point_cloud_dims_max = batch_data_label["point_cloud_dims_max"]
            gt_bounding_boxes = batch_data_label["gt_box_corners"]

            batch_pred_map_cls = parse_predictions(
                predicted_boxes,
                sem_cls_probs,
                objectness_probs,
                point_cloud,
                config_dict,
            )

            # Create a directory for each element in the batch
            for i in range(point_cloud.size(0)):
                element_dir = os.path.join(name_folder, f"element_{batch_idx}_{i}")
                os.makedirs(element_dir, exist_ok=True)

                GT = os.path.join(element_dir, "GT")
                if os.path.exists(GT):
                    shutil.rmtree(GT)
                os.makedirs(GT)

                # Save bounding boxes
                for j, pred in enumerate(batch_pred_map_cls[i]):
                    _, box_params, _ = pred
                    bbox_file_path = os.path.join(element_dir, f"bbox_{j}.txt")
                    with open(bbox_file_path, "w") as bbox_file:
                        for corner in box_params:
                            corner_str = " ".join(map(str, corner))
                            bbox_file.write(f"{corner_str}\n")

                # Save ground truth bounding boxes
                for j, gt in enumerate(gt_bounding_boxes[i]):
                    box_params = gt
                    is_all_zeros = torch.all(box_params == 0)

                    if not is_all_zeros:
                        box_params = box_params.cpu().numpy()
                        bbox_file_path = os.path.join(GT, f"gt_bbox_{j}.txt")
                        with open(bbox_file_path, "w") as bbox_file:
                            for corner in box_params:
                                corner_str = " ".join(map(str, corner))
                                bbox_file.write(f"{corner_str}\n")

                # Save point cloud data
                pc_file_path = os.path.join(element_dir, "point_cloud.txt")
                with open(pc_file_path, "w") as pc_file:
                    pc_data = point_cloud[i].cpu().numpy()

                    pc_data = flip_axis_to_depth(pc_data)
                    pc_data[:, 2] *= -1
                    pc_data[:, 1] *= -1

                    pc_data[:, 3:] = pc_data[:, 3:] * 255

                    for point in pc_data:
                        pc_file.write(" ".join(map(str, point)) + "\n")

                pc_file_path = os.path.join(element_dir, "point_cloud_modelo.txt")
                with open(pc_file_path, "w") as pc_file:
                    pc_data = point_cloud_modelo[i].cpu()

                    pc_data = flip_axis_to_depth(pc_data)
                    pc_data[:, 2] *= -1
                    pc_data[:, 1] *= -1

                    for point in pc_data:
                        pc_file.write(" ".join(map(str, point)) + "\n")
