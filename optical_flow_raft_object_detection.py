import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T

from object_detection_utils import (get_directory_names, list_image_files,
                                    preprocess)
from optical_flow_raft_utils import (Args, find_bounding_boxes, inference,
                                     load_model, process_image)

sys.path.append("RAFT/core")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


################################################
########### AVENUE DATASET SETUP ###############
################################################

## TRAIN SET ##
train_dir = "./datasets/Avenue Dataset/train__/"
train_video_dirs = get_directory_names(train_dir)
train_video_paths = []
total_frames = 0
for dir in train_video_dirs:
    cur_dir = os.path.join(train_dir, dir)
    train_video_paths.append(cur_dir)
    jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
    total_frames += len(jpg_files)
    print(cur_dir, len(jpg_files))
print(f"Total Frames: {total_frames}")

## TEST SET ##
test_dir = "./datasets/Avenue Dataset/test__/"
test_video_dirs = get_directory_names(test_dir)
test_video_paths = []
total_frames = 0
for dir in test_video_dirs:
    cur_dir = os.path.join(test_dir, dir)
    test_video_paths.append(cur_dir)
    jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
    total_frames += len(jpg_files)
    print(cur_dir, len(jpg_files))
print(f"Total Frames: {total_frames}")

## PREDS, OBJECTS, IMAGES
preds_path = "./datasets/Avenue Dataset/predictions/"
train_preds_path = os.path.join(preds_path, "train/")
test_preds_path = os.path.join(preds_path, "test/")
print(train_preds_path, test_preds_path)

objects_path = "./datasets/Avenue Dataset/objects/"
train_objects_path = os.path.join(objects_path, "train/")
test_objects_path = os.path.join(objects_path, "test/")
print(train_objects_path, test_objects_path)

image_names = [img.split("/")[-1] for img in list_image_files(test_video_paths[-1])]
print(image_names[:5])

##############################################

################################################
######## SHANGHAITECH DATASET SETUP ############
################################################
# train_dir = "./datasets/shanghaitech/training/frames/"

# ## TRAIN SET
# train_video_dirs = get_directory_names(train_dir)
# print(train_video_dirs[:5])
# train_video_paths = []
# total_frames = 0

# for dir in train_video_dirs:
#     cur_dir = os.path.join(train_dir, dir)
#     train_video_paths.append(cur_dir)
#     jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
#     total_frames += len(jpg_files)
#     print(cur_dir, len(jpg_files))
# print(f"Total Frames: {total_frames}")  # 274515

# test_dir = "./datasets/shanghaitech/testing/frames/"

# ## TEST SET
# test_video_dirs = get_directory_names(test_dir)
# print(test_video_dirs[:5])
# test_video_paths = []
# total_frames = 0

# for dir in test_video_dirs:
#     cur_dir = os.path.join(test_dir, dir)
#     test_video_paths.append(cur_dir)
#     jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
#     total_frames += len(jpg_files)
#     print(cur_dir, len(jpg_files))
# print(f"Total Frames: {total_frames}")  # 40791

# ## PREDS, OBJECTS, IMAGES
# preds_path = "./datasets/shanghaitech/predictions/"
# train_preds_path = os.path.join(preds_path, "train/")
# test_preds_path = os.path.join(preds_path, "test/")
# # create_directories(train_preds_path, train_video_dirs)

# objects_path = "./datasets/shanghaitech/objects/"
# train_objects_path = os.path.join(objects_path, "train/")
# test_objects_path = os.path.join(objects_path, "test/")
# # create_directories(train_objects_path, train_video_dirs)

# image_names = [img.split("/")[-1] for img in list_image_files(test_video_paths[-1])]
# print(image_names[:5])

#################################################

################################################
######## UBNORMAL DATASET SETUP ###############
################################################

# test_dir = "./datasets/UBNormal/test/"
# normal_test_dir = os.path.join(test_dir, "test_normal_frames/")
# abnormal_test_dir = os.path.join(test_dir, "test_abnormal_frames/")

# test_video_dirs = get_directory_names(normal_test_dir)
# ab_test_dirs = get_directory_names(abnormal_test_dir)


# test_video_paths = []
# total_frames = 0
# for dir in test_video_dirs:
#     cur_dir = os.path.join(test_dir, dir)
#     test_video_paths.append(cur_dir)
#     jpg_files = [f for f in os.listdir(cur_dir) if f.endswith('.jpg')]
#     total_frames += len(jpg_files)
#     print(cur_dir, len(jpg_files))

# test_video_paths = []
# total_frames = 0
# for dir in test_video_dirs:
#     cur_dir = os.path.join(normal_test_dir, dir)
#     test_video_paths.append(cur_dir)
#     jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
#     total_frames += len(jpg_files)
#     print(cur_dir, len(jpg_files))

# for dir in ab_test_dirs:
#     cur_dir = os.path.join(abnormal_test_dir, dir)
#     test_video_paths.append(cur_dir)
#     jpg_files = [f for f in os.listdir(cur_dir) if f.endswith(".jpg")]
#     total_frames += len(jpg_files)
#     print(cur_dir, len(jpg_files))

# test_video_dirs.extend(ab_test_dirs)

# print(len(test_video_paths))  # 211
# print(f"Total Frames: {total_frames}")  # 92640
# video_names = test_video_dirs

#
############# INFERENCE PARAMS ###############

############ Avenue
# pixel_threshold=200 # 200
# im_size = (640, 360)
# min_area_threshold = 200
############ Shanghaitech or UBNORMAL
pixel_threshold = 0
# im_size = (856, 480)
min_area_threshold = 0
###################################

transform = T.Compose(
    [
        # T.Resize(im_size),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)
bbox_temp = {}
video_names = [video.split("/")[-1] for video in test_video_paths]
print(video_names[:3])
obj_dect_opt_flow = {}
look_ahead = 3

# Define and Load Model
model = load_model("RAFT/models/raft-sintel.pth", args=Args())
model = model.to(device)
model.eval()


def inference_loop():
    # iterate every 2 frames in test set
    for i, video_path in enumerate(test_video_paths):
        print(video_path)
        image_names = [
            img.split("/")[-1] for img in list_image_files(test_video_paths[i])
        ]
        bbox_temp = {}

        frame_idx = 0
        # Iterate every 2 frames
        for frame_idx in range(0, len(image_names), 2):
            # Re-init frame variables
            image_fr1 = image_fr2 = None

            # Get frame at frame_idx
            full_image_path_fr1 = os.path.join(
                test_video_paths[i], image_names[frame_idx]
            )
            image_tensor_fr1, image_fr1 = preprocess(
                full_image_path_fr1, transform=transform
            )

            # Try to get frame at frame_idx + lookahead
            try:
                full_image_path_fr2 = os.path.join(
                    test_video_paths[i], image_names[frame_idx + look_ahead]
                )
                image_tensor_fr2, image_fr2 = preprocess(
                    full_image_path_fr2, transform=transform
                )
            except:
                pass

            # We couldn't get desired frame, it means we're close to the end
            if image_fr2 is None:
                # Get frame at frame_idx + 3 (new: look-ahead)
                if frame_idx + look_ahead > len(image_names) - 1:
                    # Get frame at frame_idx + 2
                    full_image_path_fr2 = os.path.join(
                        test_video_paths[i], image_names[len(image_names) - 1]
                    )
                    image_tensor_fr2, image_fr2 = preprocess(
                        full_image_path_fr2, transform=transform
                    )

            # This code may now be redundant
            if image_fr2 is None:
                try:
                    full_image_path_fr2 = os.path.join(
                        test_video_paths[i], image_names[frame_idx + 1]
                    )
                    image_tensor_fr2, image_fr2 = preprocess(
                        full_image_path_fr2, transform=transform
                    )
                except:
                    pass

                # if frame at frame_idx + 1 is None, it means we're at the last frame in the video
                if image_fr2 is None:
                    # Set bounding boxes for current frame as previous frame and break
                    bbox_temp[frame_idx] = bounding_boxes
                    obj_dect_opt_flow[video_names[i]] = bbox_temp

                    continue

            # Get optical flow for frame frame_idx, and frame_idx+2
            _, flow_up = inference(
                model,
                np.array(image_fr1),
                np.array(image_fr2),
                iters=20,
                device="cuda",
                test_mode=True,
            )
            # flow_iters = inference(model, np.array(image_fr1), np.array(image_fr2), device='cuda', iters=50, test_mode=True)

            # Process all images
            np_seg_map1 = process_image(flow_up.squeeze(0)[0])
            # np_seg_map1 = process_image(flow_iters[-1].squeeze(0)[0])
            # np_seg_map2 = process_image(flow_up.squeeze(0)[1])
            np_orig_image1 = np.array(image_fr1)
            # np_orig_image2 = np.array(image_fr2)

            # Get bounding boxes for frame i
            bounding_boxes = find_bounding_boxes(
                np_orig_image1,
                np_seg_map1,
                threshold=pixel_threshold,
                min_area_threshold=min_area_threshold,
                debug=False,
            )

            bbox_temp[frame_idx] = bounding_boxes
            bbox_temp[frame_idx + 1] = bounding_boxes

        # Assign same bounding boxes to frame i+1 as in frame i
        obj_dect_opt_flow[video_names[i]] = bbox_temp

    return obj_dect_opt_flow


def main():
    obj_dect_opt_flow = inference_loop()
    obj_dect_opt_flow_name = (
        f"optical_flow_raft_ubnormal_thresh_0_minarea_0_iterations_20_lookahead_2"
    )
    with open(obj_dect_opt_flow_name, "wb") as file:
        pickle.dump(obj_dect_opt_flow, file)


if __name__ == "__main__":
    main()
