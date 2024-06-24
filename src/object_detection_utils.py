import glob
import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as ops
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

device = "cuda" if torch.cuda.is_available() else "cpu"

"""Image Processing"""


# Load and preprocess the image
def preprocess(image_path, transform):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor, image


"""Image Visualisation"""


def imshow(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    # img = std * img + mean  # unnormalize
    # img = np.clip(img, 0, 1)  # clip any values outside the range [0, 1]
    plt.imshow(img)
    plt.show()


def display_segmentation_map(image_path):
    """
    Displays a segmentation map given the path of the image.

    Parameters:
    - image_path: The file path to the segmentation map image.
    """
    # Load the image from the specified path
    segmentation_map = mpimg.imread(image_path)

    # Display the image
    plt.imshow(segmentation_map)
    plt.title("Segmentation Map")
    plt.axis("off")  # Hide the axes
    plt.show()

    return segmentation_map


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


# COCO classes
# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=3,
                alpha=0.8,
            )
        )
        cl = p.argmax()
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # ax.text(xmin, ymin, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis("off")
    plt.show()


def _plot_result_single_image_in_batch(pil_img, prob, boxes, im_size=(500, 500)):
    # Calculate scale factors for width and height
    orig_width, orig_height = pil_img.size
    scale_x = orig_width / im_size[0]
    scale_y = orig_height / im_size[1]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        # Scale box coordinates back to original image size
        xmin, xmax = xmin * scale_x, xmax * scale_x
        ymin, ymax = ymin * scale_y, ymax * scale_y

        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=3,
                alpha=0.8,
            )
        )
        cl = p.argmax()
        # Optionally, add text for each box
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # ax.text(xmin, ymin, text, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis("off")
    plt.show()


def plot_batch_detections(images, detections_batch):
    """
    Plots the results for a batch of images and their detections.

    :param images: List of PIL images.
    :param detections_batch: List of detections for each image in the batch.
                             Each detection should be a tuple of (probabilities, boxes).
    """
    for img, detections in zip(images, detections_batch):
        prob, boxes = detections
        _plot_result_single_image_in_batch(img, prob, boxes)


def plot_results_avenue(
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

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
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


def plot_latent_representations(
    tensor: torch.tensor, title: str = "3D Scatter Plot of Latent Space"
) -> None:
    # Flattening the tensor except for the batch dimension
    flattened_tensor = tensor.view(tensor.size(0), -1).detach().cpu()

    x = flattened_tensor[:, 0].numpy()
    y = flattened_tensor[:, 1].numpy()
    z = flattened_tensor[:, 2].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)

    ax.set_xlabel("512-dim (1st)")
    ax.set_ylabel("2-dim (2nd)")
    ax.set_zlabel("2-dim (3rd)")
    ax.set_title(title)

    plt.show()


"""Object Dection in Images"""


def detect(image_tensor, model, im_size, threshold=0.4):
    # Move the NestedTensor to the device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Run the model
        outputs = model(image_tensor)

    # keep only predictions with 0.7+ confidence
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    # print(probas)
    keep = probas.max(-1).values > threshold

    # Extract scores and bounding boxes
    scores = probas[keep].max(-1).values
    bboxes = outputs["pred_boxes"][0, keep]

    # Apply NMS
    # keep_boxes = ops.nms(bboxes, scores, iou_threshold=0.8)  # Adjust the iou_threshold as necessary

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], im_size)
    return probas[keep], bboxes_scaled


def batch_detect(
    nested_tensor, model, device, iou_threshold=0.85, object_keep_probab=0.3
):
    # Move the NestedTensor to the device
    nested_tensor = nested_tensor.to(device)

    with torch.no_grad():
        # Run the model
        outputs = model(nested_tensor)

    batch_detections = []

    # Extract tensor from NestedTensor
    tensor = nested_tensor.tensors
    for i in range(tensor.shape[0]):
        probas = outputs["pred_logits"][i].softmax(-1)[:, :-1]
        keep = probas.max(-1).values > object_keep_probab

        # Extract scores and bounding boxes
        scores = probas[keep].max(-1).values
        bboxes = outputs["pred_boxes"][i, keep]

        # Apply NMS
        keep_boxes = ops.nms(
            bboxes, scores, iou_threshold=iou_threshold
        )  # Adjust iou_threshold as needed

        # Rescale bounding boxes (assuming rescale_bboxes function handles this)
        bboxes_scaled = rescale_bboxes(
            bboxes[keep_boxes], nested_tensor.mask[i].shape[-2:]
        )

        batch_detections.append((probas[keep][keep_boxes], bboxes_scaled))

    return batch_detections


"""File Manipulation"""


def list_files(directory, file_types=["*.pkl"]):
    files = []
    for file_type in file_types:
        files.extend(glob.glob(os.path.join(directory, file_type)))
    return files


def list_image_files(directory, file_types=["*.jpg", "*.jpeg", "*.png"]):
    return list_files(directory, file_types=file_types)


def get_directory_names(parent_directory):
    directory_names = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    return directory_names


def load_images_from_folder(folder):
    """
    Load all images in a folder into a list of PIL Image objects.

    :param folder: Path to the directory containing image files.
    :return: List of PIL Image objects.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                images.append(img.copy())
        except IOError:
            # This will skip any files that aren't valid images
            print(
                f"Skipping file {filename}, unable to open or it's not an image file."
            )
    return images


def read_directory_names_from_file(file_path):
    """
    Reads a text file and returns a list of directory names.

    Parameters:
    - file_path: Full path to the text file containing directory names.

    Returns:
    A list of directory names.
    """
    directory_names = []

    # Open the file and read line by line
    with open(file_path, "r") as file:
        for line in file:
            # Strip any leading/trailing whitespace (including newlines)
            directory_name = line.strip()
            directory_names.append(directory_name)

    return directory_names


def save_cropped_images(cropped_images, path, image_prefix=""):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, image in enumerate(cropped_images):
        image_path = os.path.join(path, f"{image_prefix}_object_{i}.jpg")
        image.save(image_path)
        # print(f"Saved: {image_path}")


def load_artefact(path: str):
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def read_tracks(tracks_path):
    data = []
    with open(tracks_path, "r") as file:
        for line in file:
            # Split the line by comma, convert each part first to float, then to int
            numbers = [int(float(number)) for number in line.strip().split(",")]
            # Append the list of numbers to the data list
            data.append(numbers)

    return data
