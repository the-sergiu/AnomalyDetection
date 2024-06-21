import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, ImageFont

from object_detection_utils import COLORS
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

sys.path.append("RAFT/core")



# class to interface with RAFT
class Args:
    def __init__(
        self,
        model="",
        path="",
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    ):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    """ Sketchy hack to pretend to iterate through the class objects """

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


def plot_results(
    pil_img,
    prob,
    boxes,
    im_size=(640, 360),
    display_img=True,
    save_path=None,
    crop_objects=False,
):
    orig_width, orig_height = pil_img.size
    scale_x = orig_width / im_size[0]
    scale_y = orig_height / im_size[1]

    # Create a copy of the image for cropping
    pil_img_copy = pil_img.copy()

    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()  # Can be changed to another font

    cropped_images = []  # List to hold cropped objects

    for (xmin, ymin, xmax, ymax), c in zip(boxes, COLORS * 100):
        xmin, xmax = xmin * scale_x, xmax * scale_x
        ymin, ymax = ymin * scale_y, ymax * scale_y

        color = tuple(int(255 * x) for x in c)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        if crop_objects:
            # Cropping from the unaltered image copy
            cropped_obj = pil_img_copy.crop((xmin, ymin, xmax, ymax))
            cropped_images.append(cropped_obj)

        # Optionally add class text
        # cl = p.argmax()
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # draw.text((xmin, ymin), text, fill=color, font=font)

    if display_img:
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        plt.axis("off")
        plt.show()

    if save_path:
        pil_img.save(save_path)

    return pil_img, cropped_images


def imshow(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    # img = std * img + mean  # unnormalize
    # img = np.clip(img, 0, 1)  # clip any values outside the range [0, 1]
    plt.imshow(img)
    plt.show()


# Function to preprocess and return numpy array from tensor
def process_image(tensor):
    np_image = tensor.squeeze(0).detach().cpu().numpy()
    if np_image.ndim == 3 and np_image.shape[0] in [3, 4]:  # for RGB or RGBA
        np_image = np_image.transpose(1, 2, 0)
    return np_image


def numpy_to_cuda(img, device="cuda"):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def find_bounding_boxes(
    orig_image, seg_map, threshold=200, min_area_threshold=200, debug=False
):
    """
    Find bounding boxes of objects in an image based on a threshold.

    Parameters:
    image (numpy array): The input image.
    threshold (int): Threshold value for binary segmentation.
    min_area_threshold (int): Minimum area threshold for considering an object.
    debug (bool): If True, show the image with detected objects.

    Returns:
    List of bounding boxes in the format [xmin, ymin, xmax, ymax].
    """
    drawable = orig_image.copy()
    image = seg_map.astype(np.uint8)
    # Convert the image to grayscale if it's colored
    gray_image = (
        cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )

    # Apply a binary threshold to the image
    _, binary_image = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)

    # # Define kernel for morphological operations
    # kernel = np.ones((7, 7), np.uint8)

    # # Apply erosion and dilation
    # binary_image = cv.erode(binary_image, kernel, iterations=1)
    # binary_image = cv.dilate(binary_image, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    bounding_boxes = []
    for i in range(1, num_labels):  # Skip label 0 as it's the background
        x, y, w, h, area = stats[i]
        # Check if the area of the blob is greater than the minimum threshold
        if area > min_area_threshold:
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            bounding_boxes.append([xmin, ymin, xmax, ymax])

            if debug:
                # Draw a rectangle and centroid for visualization
                cv.rectangle(drawable, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                cx, cy = centroids[i]
                cv.circle(drawable, (int(cx), int(cy)), 5, (0, 255, 0), -1)

    if debug:
        # Show the image with detected objects
        # plt.imshow(drawable)

        # Create a figure with 2x2 grid of axes
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot each image on its respective axes
        axs[0].imshow(binary_image)
        axs[0].axis("off")  # Turn off axis
        axs[0].set_title("Binary Image (raw)")

        axs[1].imshow(drawable)
        axs[1].axis("off")
        axs[1].set_title("Original image with bounding boxes")

    return bounding_boxes


def load_model(weights_path, args):
    """Loads model to CUDA only"""
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cuda")
    return model


def inference(
    model,
    frame1,
    frame2,
    device="cuda",
    pad_mode="sintel",
    iters=12,
    flow_init=None,
    upsample=True,
    test_mode=True,
):
    model.eval()
    with torch.no_grad():
        # preprocess
        frame1 = numpy_to_cuda(frame1, device)
        frame2 = numpy_to_cuda(frame2, device)

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow
        if test_mode:
            flow_low, flow_up = model(
                frame1,
                frame2,
                iters=iters,
                flow_init=flow_init,
                upsample=upsample,
                test_mode=test_mode,
            )
            return flow_low, flow_up

        else:
            flow_iters = model(
                frame1,
                frame2,
                iters=iters,
                flow_init=flow_init,
                upsample=upsample,
                test_mode=test_mode,
            )

            return flow_iters


def get_viz(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    return flow_viz.flow_to_image(flo)


def display_flow_low_flow_up(frame1, frame2, flow_low, flow_up) -> None:
    # Process all images
    np_seg_map1 = process_image(flow_up.squeeze(0)[0])
    np_seg_map2 = process_image(flow_up.squeeze(0)[1])
    np_orig_image1 = np.array(frame1)
    np_orig_image2 = np.array(frame2)

    # Create a figure with 2x2 grid of axes
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot each image on its respective axes
    axs[0, 0].imshow(np_seg_map1)
    axs[0, 0].axis("off")  # Turn off axis
    axs[0, 0].set_title("Flow 1")

    axs[0, 1].imshow(np_seg_map2)
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Flow 2")

    axs[1, 0].imshow(np_orig_image1)
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Original Image 1")

    axs[1, 1].imshow(np_orig_image2)
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Original Image 2")

    # Adjust the spacing
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.5, bottom=0.1, hspace=0.2, wspace=0.2
    )

    # Display the figure
    plt.show()

    return np_orig_image1, np_seg_map1, np_orig_image2, np_seg_map2


def filter_bounding_boxes_by_area(bounding_boxes, min_area):
    """
    Filter bounding boxes by a minimum area threshold.

    Parameters:
    bounding_boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax].
    min_area (int): Minimum area threshold.

    Returns:
    List of bounding boxes that have an area greater than or equal to the min_area.
    """
    filtered_boxes = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        area = (xmax - xmin) * (ymax - ymin)
        if area >= min_area:
            filtered_boxes.append(box)
    return filtered_boxes


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union of two bounding boxes.
    Each box is defined as [xmin, ymin, xmax, ymax].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)

    area_inter = width_inter * height_inter
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    area_union = area_box1 + area_box2 - area_inter

    iou = area_inter / area_union if area_union != 0 else 0
    # print(iou)
    return iou


def merge_bounding_boxes(list1, list2, iou_threshold):
    """
    Merge two lists of bounding boxes based on IoU threshold.
    """
    for box2 in list2:
        if all(compute_iou(box2, box1) < iou_threshold for box1 in list1):
            list1.append(box2)
    return list1
