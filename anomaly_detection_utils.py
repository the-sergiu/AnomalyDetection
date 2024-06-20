import os
import sys
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from evaluation.merge_tracks import ContinuousTrack
from PIL import Image
from sklearn import metrics
from torch.nn import functional as F
from torch.utils.data import Dataset

from object_detection_utils import imshow, list_image_files

sys.path.append("./abnorm_event_detect/evaluation/")
os.chdir("abnorm_event_detect")


os.chdir("..")


"""Adversarial Training Related"""


class GaussianNoiseDataset(Dataset):
    def __init__(self, length, size, mu=0.0, scale=1.0):
        self.length = length
        self.size = size
        self.mu = mu
        self.scale = scale

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noise = np.random.normal(loc=self.mu, scale=self.scale, size=self.size)
        return torch.tensor(noise, dtype=torch.float32)


"""Anomaly Score Computation"""


class TrackState(Enum):
    CREATED = "created"
    UPDATED = "updated"
    CLOSED = "closed"


class Track:
    def __init__(self, start_idx=0, end_idx=None, mask=0, video_name=""):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.bboxes = {}
        self.mask = mask
        self.state = TrackState.CREATED
        self.video_name = video_name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class AnomalyDetection:
    def __init__(self, frame_idx, bbox, score, video_name, track_id=-1):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.score = score
        self.video_name = video_name
        self.track_id = track_id

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_iou(pred_anomaly, gt_anomalies_per_frame):
    max_iou = 0
    idx = -1
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if max_iou < iou:
            max_iou = iou
            idx = index

    return max_iou, idx


def get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta):
    indices = []
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if iou >= beta:
            indices.append(index)

    return indices


def compute_tbdr(gt_tracks, num_matched_detections_per_track, alpha):
    percentages = np.array(
        [x / len(y.bboxes) for x, y in zip(num_matched_detections_per_track, gt_tracks)]
    )
    return np.sum(percentages >= alpha) / len(num_matched_detections_per_track)


def compute_fpr_rbdr(
    pred_anomalies_detected: List[AnomalyDetection],
    gt_anomalies: List[AnomalyDetection],
    all_gt_tracks,
    num_frames,
    num_tracks,
    alpha=0.1,
    beta=0.1,
):
    num_matched_detections_per_track = [0] * num_tracks

    # TODO: add pixel level IOU
    num_detected_anomalies = len(pred_anomalies_detected)
    gt_anomaly_video_per_frame_dict = {}
    found_gt_anomaly_video_per_frame_dict = {}

    for anomaly in gt_anomalies:
        anomalies_per_frame = gt_anomaly_video_per_frame_dict.get(
            (anomaly.video_name, anomaly.frame_idx), None
        )
        if anomalies_per_frame is None:
            gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)] = [
                anomaly
            ]
            found_gt_anomaly_video_per_frame_dict[
                (anomaly.video_name, anomaly.frame_idx)
            ] = [0]
        else:
            gt_anomaly_video_per_frame_dict[
                (anomaly.video_name, anomaly.frame_idx)
            ].append(anomaly)
            found_gt_anomaly_video_per_frame_dict[
                (anomaly.video_name, anomaly.frame_idx)
            ].append(0)

    tp = np.zeros(num_detected_anomalies)
    fp = np.zeros(num_detected_anomalies)
    tbdr = np.zeros(num_detected_anomalies)
    remove_idx = []
    pred_anomalies_detected.sort(
        key=lambda anomaly_detection: anomaly_detection.score, reverse=True
    )
    for idx, pred_anomaly in enumerate(pred_anomalies_detected):
        gt_anomalies_per_frame = gt_anomaly_video_per_frame_dict.get(
            (pred_anomaly.video_name, pred_anomaly.frame_idx), None
        )

        if gt_anomalies_per_frame is None:
            fp[idx] = 1
        else:
            matching_gt_bboxes_indices = get_matching_gt_indices(
                pred_anomaly, gt_anomalies_per_frame, beta
            )
            if len(matching_gt_bboxes_indices) > 0:
                non_matched_indices = []
                for matched_ind in matching_gt_bboxes_indices:
                    if (
                        found_gt_anomaly_video_per_frame_dict.get(
                            (pred_anomaly.video_name, pred_anomaly.frame_idx)
                        )[matched_ind]
                        == 0
                    ):
                        non_matched_indices.append(matched_ind)
                        found_gt_anomaly_video_per_frame_dict.get(
                            (pred_anomaly.video_name, pred_anomaly.frame_idx)
                        )[matched_ind] = 1
                        num_matched_detections_per_track[
                            gt_anomalies_per_frame[matched_ind].track_id
                        ] += 1

                tp[idx] = len(non_matched_indices)

            else:
                fp[idx] = 1

        tbdr[idx] = compute_tbdr(all_gt_tracks, num_matched_detections_per_track, alpha)

    cum_false_positive = np.cumsum(fp)
    cum_true_positive = np.cumsum(tp)
    # add the point (0, 0) for each vector
    cum_false_positive = np.concatenate(([0], cum_false_positive))
    cum_true_positive = np.concatenate(([0], cum_true_positive))
    tbdr = np.concatenate(([0], tbdr))

    rbdr = cum_true_positive / len(gt_anomalies)
    fpr = cum_false_positive / num_frames

    idx_1 = np.where(fpr <= 1)[0][-1] + 1

    if fpr[idx_1 - 1] != 1:
        print("fpr does not reach 1")
        rbdr = np.insert(rbdr, idx_1, rbdr[idx_1 - 1])
        tbdr = np.insert(tbdr, idx_1, tbdr[idx_1 - 1])
        fpr = np.insert(fpr, idx_1, 1)
        idx_1 += 1

    tbdc = metrics.auc(fpr[:idx_1], tbdr[:idx_1])
    rbdc = metrics.auc(fpr[:idx_1], rbdr[:idx_1])

    print("tbdc = " + str(tbdc))
    print("rbdc = " + str(rbdc))
    return rbdc, tbdc

    print(tbdr[idx_1 - 1], rbdr[idx_1 - 1])
    plt.plot(fpr, rbdr, "-")
    plt.xlabel("FPR")
    plt.ylabel("RBDR")
    plt.show()


def get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta):
    indices = []
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if iou >= beta:
            indices.append(index)

    return indices


# range 302, mu 25
# UBNormal: 452, Mu: 201
def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    # mu = np.mean(support)
    filter = (
        1.0
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    )
    return filter


def filt(input, dim=9, range=302, mu=25):
    filter_3d = np.ones((dim, dim, dim)) / (dim**3)
    filter_2d = gaussian_filter(np.arange(1, range), mu)

    frame_scores = input  # This works
    # frame_scores = convolve(input, filter_3d)
    # frame_scores = frame_scores.max((1, 2))

    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, "valid")
    return frame_scores


def process_current_vid_preds(pred: np.array, range=302, mu=25, norm=False):
    pred = np.nan_to_num(pred, nan=0.0)
    pred = filt(pred, range=range, mu=mu)
    if norm:
        pred = (pred - np.min(pred)) / (
            np.max(pred) - np.min(pred)
        )  # cu si fara, sa vedem ce obtinem
    return pred


def read_txt_to_numpy_array(file_path):
    try:
        data = np.loadtxt(file_path)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def compute_macro_auc(anomaly_scores_dict: dict, labels_dict: dict):
    aucs = []
    filtered_preds = []
    filtered_labels = []

    for vid_name in anomaly_scores_dict:
        # print(len(anomaly_scores_dict[vid_name]))
        pred = np.array(list(score for score in anomaly_scores_dict[vid_name].values()))
        pred = process_current_vid_preds(pred)
        filtered_preds.append(pred)

        lbl = labels_dict[vid_name]
        filtered_labels.append(lbl)

        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])

        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)
    print(macro_auc)

    return macro_auc, filtered_preds, filtered_labels


def compute_micro_auc(filtered_preds, filtered_labels):
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)
    print(micro_auc)

    return micro_auc


autoencoder_transform = T.Compose(
    [T.Resize((64, 64)), T.ToTensor(), T.Normalize(0.5, 0.5)]
)


def compute_anomaly_reconstruction_scores(
    autoencoder,
    obj_dect_dict,
    test_video_paths,
    video_names,
    device="cuda",
    verbose=True,
):
    autoencoder.eval()
    all_pred_ano = []
    anomaly_scores_dict = {}
    for i, video_path in enumerate(test_video_paths):
        if verbose:
            print(video_path)
        image_names = [
            img.split("/")[-1] for img in list_image_files(test_video_paths[i])
        ]

        # Get dict containing {frame_idx: bounding boxes} for current video
        bbox_temp = obj_dect_dict[video_names[i]]

        # For each video, all frames will have an associated anomaly score given by the max MLE loss on all objects in that frame
        anomaly_scores_dict[video_names[i]] = {}

        # Iterate through
        for frame_idx, image_name in zip(bbox_temp, image_names):
            # Get full path to frame/image
            full_image_path = os.path.join(test_video_paths[i], image_name)

            image = Image.open(full_image_path)

            # Get list of bounding boxes
            boxes = bbox_temp[frame_idx]

            current_frame_max_anomaly_score = float("-inf")

            # Initialize a numpy array of zerios, size of image
            frame_anomaly_map = np.zeros(
                image.size[::-1]
            )  # Image.size returns (width, height)

            # Go through all bounding boxes of that frame, and crop the objects
            for bbox in boxes:
                cropped_obj = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Plot the cropped object for debugging
                # cropped_obj.show()  # This will display the cropped image

                # Transform cropped obj to desired 64 * 64 shape
                transformed_obj = autoencoder_transform(cropped_obj)

                # Ensure the transformed object is in the right shape for the model
                transformed_obj = transformed_obj.unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    # Forward pass through the encoder and then the decoder
                    # latent_representation = encoder(transformed_obj.to(device))[-1]  # Assuming the last output is the latent representation
                    # reconstructed_img = decoder(latent_representation)
                    reconstructed_img = autoencoder(transformed_obj.to(device))

                # Ensure cropped_obj is a tensor and in the correct shape for loss calculation
                cropped_obj_tensor = autoencoder_transform(cropped_obj)
                cropped_obj_tensor = cropped_obj_tensor.unsqueeze(
                    0
                )  # Add batch dimension

                # Compute the reconstruction loss
                score = F.mse_loss(reconstructed_img, cropped_obj_tensor.to(device))

                # Given a bounding box, add the score obtained above to the corresponding positions numpy array
                # Use bounding box coordinates
                frame_anomaly_map[
                    int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                ] += score.item()
                try:
                    anomaly_score = frame_anomaly_map[
                        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                    ].max()
                    all_pred_ano.append(
                        AnomalyDetection(
                            frame_idx, bbox, anomaly_score, video_names[i], track_id=-1
                        )
                    )
                except:
                    pass

            current_frame_max_anomaly_score = np.max(frame_anomaly_map)
            anomaly_scores_dict[video_names[i]][
                frame_idx
            ] = current_frame_max_anomaly_score

    return anomaly_scores_dict, all_pred_ano


def compute_loss_on_dataloader(
    autoencoder, dataloader: torch.utils.data.DataLoader, device: str = "cpu"
) -> None:
    total_loss = 0.0
    total_batches = 0

    # No gradients needed for evaluation, which saves memory and computations
    with torch.no_grad():
        for images, _ in dataloader:  # labels are not needed for loss computation
            # Move images to the device your model is on
            images = images.to(device)

            with torch.no_grad():
                # Forward pass through the encoder and then the decoder
                reconstructed_imgs = autoencoder(images)

            # Compute the reconstruction loss
            loss = F.mse_loss(reconstructed_imgs, images)

            # Accumulate the loss
            total_loss += loss.item()
            total_batches += 1

    # Calculate the average loss over all batches
    return total_loss / total_batches


def plot_images_vs_reconstructed_images(
    images,
    reconstructed_imgs,
    first_batch_title: str = "Original Images",
    second_batch_title="Reconstructed Images",
) -> None:
    # Move images back to cpu for visualization
    images = images.cpu()
    reconstructed_imgs = reconstructed_imgs.cpu()

    # Display original and reconstructed images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title(first_batch_title)
    imshow(vutils.make_grid(images, padding=2, normalize=True))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 2)
    plt.title(second_batch_title)
    imshow(vutils.make_grid(reconstructed_imgs, padding=2, normalize=True))
