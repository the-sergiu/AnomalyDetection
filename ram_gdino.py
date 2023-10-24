import os
import random
import re

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
import torchvision.transforms as TS
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image, ImageDraw, ImageFont
from ram import inference_ram
from ram import inference_tag2text
from ram.models import ram
from ram.models import tag2text
from segment_anything import SamPredictor, build_sam
from typing import Tuple, List, Optional, Union, Any

# Model Paths
config_file = "./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ram_checkpoint = "./pretrained/ram_swin_large_14m.pth"
tag2text_checkpoint = "./pretrained/tag2text_swin_14m.pth"
grounded_checkpoint = "./Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
sam_checkpoint = "./Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Threshold Values
box_threshold = 0.25
text_threshold = 0.2
iou_threshold = 0.5


def load_model(model_config_path: str, model_checkpoint_path: str, device: str) -> Any:
    """
    Load a model with the given configuration and checkpoint paths.

    Returns:
        Any: Loaded model.
    """
    # Load configuration from file
    args = SLConfig.fromfile(model_config_path)
    args.device = device

    # Build model using the loaded configuration
    model = build_model(args)

    # Load checkpoint into the model
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)  # Log loading results
    model.eval()

    return model


def get_grounding_output(
    model: torch.nn.Module,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Performs grounding on the given image using the specified model and caption.

    Args:
        model (torch.nn.Module): The pre-trained grounding model.
        image (torch.Tensor): Image tensor of shape [C, H, W].
        caption (str): Caption describing the image.
        box_threshold (float): Threshold for filtering bounding boxes based on the logit scores.
        text_threshold (float): Threshold for creating phrases from logit positions.
        device (str, optional): Device to run the model on. Default is "cpu".

    Returns:
        torch.Tensor: Filtered bounding boxes of detected objects.
        torch.Tensor: Corresponding scores for the bounding boxes.
        List[str]: Predicted phrases associated with the bounding boxes.
    """

    # Convert caption to lowercase and append "." if not present
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    # Move model and image to the specified device
    model = model.to(device)
    image = image.to(device)

    # Inference without gradient calculation for efficiency
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # Extract prediction logits and bounding boxes from the model output
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter boxes and logits based on the box_threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt, 256
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # Tokenize the caption using the model's tokenizer
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    # Extract phrases and scores based on the logits and text_threshold
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenizer
        )
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


# Normalize and transform the image for tagging model
NORMALIZE = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TAGGING_TRANSFORM = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), NORMALIZE])
GROUNDING_TRANSFORM = T.Compose(
    [
        # T.RandomResize([800], max_size=1333),
        # Added T.Resize to fix the resized image during batch inference
        # T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def detect_objects_with_ram(
    raw_image: Image.Image,
    grounding_dino_model: torch.nn.Module,
    ram_model: torch.nn.Module,
    draw_boxes: bool = False,
    label_with_probab: bool = False,
    device: str = "cuda",
):
    # print(f"Start processing, image size {raw_image.size}")
    raw_image = raw_image.convert("RGB")

    # Tagging Model
    image = TAGGING_TRANSFORM(raw_image.resize((384, 384))).unsqueeze(0).to(device)
    res = inference_ram(image, ram_model)
    tags = res[0].strip(" ").replace("  ", " ").replace(" |", ",").lower().strip()
    if not tags.endswith("."):
        tags = tags + "."
    print("Tags: ", tags)

    # Use predefined grounding transformation for GroundingDINO model
    image, _ = GROUNDING_TRANSFORM(raw_image, None)
    # grounding_dino_model = grounding_dino_model.to(device)
    image = image.to(device)
    # Do I really need model = grounding_dino_model.to(device)?

    # boxes_filt, scores, pred_phrases = get_grounding_output(
    #     grounding_dino_model, image, tags, box_threshold, text_threshold, device=device
    # )

    outputs = grounding_dino_model(image[None], captions=[tags])

    # Extract prediction logits and bounding boxes from the model output
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    # logits = logits.to(device); boxes = boxes.to(device)

    # Filter boxes and logits based on the box_threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt, 256
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # Tokenize the caption using the model's tokenizer
    tokenizer = grounding_dino_model.tokenizer
    tokenized = tokenizer(tags)

    # Extract phrases and scores based on the logits and text_threshold
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenizer
        )
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    scores = torch.Tensor(scores)
    size = raw_image.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])  # Move to CUDA
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = (
        torchvision.ops.nms(boxes_filt, scores, iou_threshold).cpu().numpy().tolist()
    )
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = (
        [pred_phrases[idx] for idx in nms_idx]
        if label_with_probab  # If we want to display the label alongside probability
        else [re.sub(r"\(\d+\.\d+\)", "", pred_phrases[idx]).strip() for idx in nms_idx]
    )
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    # if draw_boxes:
    image_draw = ImageDraw.Draw(raw_image)
    # label2boxes = []
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)

    out_image = raw_image.convert("RGBA")

    return out_image, boxes_filt


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def predict_batch(
    model,
    images: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda",
    debug: bool = False,
):
    """
    return:
        bboxes_batch: list of tensors of shape (n, 4)
        predicts_batch: list of tensors of shape (n,)
        phrases_batch: list of list of strings of shape (n,)
        n is the number of boxes in one image
    """
    # caption = preprocess_caption(caption=caption)
    # model = model.to(device)
    image = images.to(device)
    with torch.no_grad():
        outputs = model(
            image, captions=[caption for _ in range(len(images))]
        )  # <------- I use the same caption for all the images for my use-case
    prediction_logits = outputs[
        "pred_logits"
    ].sigmoid()  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs[
        "pred_boxes"
    ]  # prediction_boxes.shape = (num_batch, nq, 4)

    # import ipdb; ipdb.set_trace()
    mask = (
        prediction_logits.max(dim=2)[0] > box_threshold
    )  # mask: torch.Size([num_batch, 256])

    bboxes_batch = []
    predicts_batch = []
    phrases_batch = []  # list of lists
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        phrases = [
            get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            ).replace(".", "")
            for logit in logits  # logit is a tensor of shape (256,) torch.Size([256])  # torch.Size([7, 256])
        ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])

        if debug:
            image_draw = ImageDraw.Draw(raw_image)
            for box, label in zip(boxes, phrases):
                draw_box(box, image_draw, label)

    return bboxes_batch, predicts_batch, phrases_batch


@torch.no_grad()
def inference(
    raw_image: Image.Image,
    specified_tags: List[str],
    do_det_seg: bool,
    tagging_model_type: str,
    tagging_model: torch.nn.Module,
    grounding_dino_model: torch.nn.Module,
    # sam_model: torch.nn.Module,
    label_with_probab: bool,
    device: str,
) -> Union[
    Tuple[str, Image.Image, List[Tuple[str, torch.Tensor]], torch.Tensor],
    Tuple[str, str, Image.Image],
]:
    """
    Perform inference on an image, tagging it, optionally running object detection and segmentation.

    Args:
        raw_image (Image.Image): Input image to process.
        specified_tags (List[str]): List of tags to consider for tagging.
        do_det_seg (bool): Whether to run detection and segmentation on the image.
        tagging_model_type (str): Type of tagging model used ("RAM" or others).
        tagging_model (torch.nn.Module): Pre-trained tagging model.
        grounding_dino_model (torch.nn.Module): Pre-trained grounding model.
        sam_model (torch.nn.Module): Pre-trained segmentation model.
        label_with_probab (bool): Whether to label the output with probabilities.
        device (str): Device to run the model on.

    Returns:
        Tuple: Depending on the operations performed and the model type, returns different types of outputs.
    """

    print(f"Start processing, image size {raw_image.size}")
    raw_image = raw_image.convert("RGB")

    image = TAGGING_TRANSFORM(raw_image.resize((384, 384))).unsqueeze(0).to(device)

    # Different handling based on tagging model type
    if tagging_model_type == "RAM":
        res = inference_ram(image, tagging_model)
        tags = res[0].strip(" ").replace("  ", " ").replace(" |", ",")
        print("Tags: ", tags)
    else:
        res = inference_tag2text(image, tagging_model, specified_tags)
        tags = res[0].strip(" ").replace("  ", " ").replace(" |", ",")
        caption = res[2]
        print(f"Tags: {tags}")
        print(f"Caption: {caption}")

    # Skip detection and segmentation if not required
    if not do_det_seg:
        if tagging_model_type == "RAM":
            return tags.replace(", ", " | "), None
        else:
            return tags.replace(", ", " | "), caption, None

    # Use predefined grounding transformation for GroundingDINO model
    image, _ = GROUNDING_TRANSFORM(raw_image, None)

    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device=device
    )
    # print("GroundingDINO finished")

    # Process image for SAM model
    # image = np.asarray(raw_image)
    # sam_model.set_image(image)

    size = raw_image.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # boxes_filt = boxes_filt.cpu()
    # Apply NMS to remove overlapped boxes
    # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]

    pred_phrases = (
        [pred_phrases[idx] for idx in nms_idx]
        if label_with_probab  # If we want to display the label alongside probability
        else [re.sub(r"\(\d+\.\d+\)", "", pred_phrases[idx]).strip() for idx in nms_idx]
    )

    # print(f"After NMS: {boxes_filt.shape[0]} boxes")

    # Commented out SAM model operations for potential future use
    # transformed_boxes = sam_model.transform.apply_boxes_torch(
    #     boxes_filt, image.shape[:2]
    # ).to(device)

    # masks, _, _ = sam_model.predict_torch(
    #     point_coords=None,
    #     point_labels=None,
    #     boxes=transformed_boxes.to(device),
    #     multimask_output=False,
    # )
    # print("SAM finished")

    # Drawing on the output image
    # mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
    # mask_draw = ImageDraw.Draw(mask_image)

    image_draw = ImageDraw.Draw(raw_image)
    # label2boxes = []
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)
        # label2boxes.append((label, box))

    out_image = raw_image.convert("RGBA")
    # out_image.alpha_composite(mask_image)

    if tagging_model_type == "RAM":
        return tags.replace(", ", " | "), out_image, pred_phrases, boxes_filt
    else:
        return tags.replace(", ", " | "), caption, out_image


def draw_mask(
    mask: np.ndarray, draw: ImageDraw.Draw, random_color: bool = False
) -> None:
    """
    Draw a mask on an image using given draw object.

    Args:
        mask (np.ndarray): Binary mask to be drawn.
        draw (ImageDraw.Draw): PIL's ImageDraw object for drawing on an image.
        random_color (bool, optional): If True, the mask is drawn in a random color.
            Otherwise, it uses a default color. Default is False.

    Returns:
        None: The function modifies the `draw` object in place.
    """

    # Choose a random or default color
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    # Get all non-zero coordinates in the mask
    nonzero_coords = np.transpose(np.nonzero(mask))

    # Draw each point of the mask
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_box(
    box: Tuple[int, int, int, int], draw: ImageDraw.Draw, label: Optional[str] = None
) -> None:
    """
    Draw a bounding box and an optional label on an image using a given draw object.

    Returns:
        None: The function modifies the `draw` object in place.
    """

    # Generate a random color for the box
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    # Define line width based on image size
    line_width = int(max(4, min(20, 0.006 * max(draw.im.size))))
    # Draw the bounding box
    draw.rectangle(
        ((box[0], box[1]), (box[2], box[3])), outline=color, width=line_width
    )

    # If there's a label, draw it
    if label:
        # Determine font path and size
        font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        font_size = int(max(12, min(60, 0.02 * max(draw.im.size))))
        font = ImageFont.truetype(font_path, size=font_size)

        # Get bounding box for the label text
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)

        # Draw the label's bounding box and the label itself
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

        draw.text((box[0], box[1]), label, font=font)


def extract_objects_from_bounding_boxes(raw_image, boxes_filt):
    """
    Extract objects from the image based on the bounding boxes.

    Args:
        raw_image (Image): The original image from which the objects will be extracted.
        boxes_filt (Tensor): Bounding boxes detected in the image.

    Returns:
        List[Image]: List of images cropped based on the bounding boxes.
    """

    # Convert raw_image to PIL Image if it isn't already
    if not isinstance(raw_image, Image.Image):
        raw_image = Image.fromarray(raw_image)

    # List to hold the cropped object images
    object_images = []

    # Loop through each bounding box and crop the corresponding area from the image
    for box in boxes_filt:
        # Convert tensor to list
        box_coords = box.tolist()

        # Convert x,y,w,h to left, upper, right, lower
        left, upper, right, lower = (
            box_coords[0],
            box_coords[1],
            box_coords[2],
            box_coords[3],
        )

        # Crop the object from the image
        cropped_img = raw_image.crop((left, upper, right, lower))

        # Append the cropped image to the list
        object_images.append(cropped_img)

    return object_images


def load_ram(device: str) -> Any:
    """
    Load the RAM model with given device settings.

    Returns:
        Any: Loaded RAM model.
    """
    ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit="swin_l")
    ram_model.eval()
    ram_model = ram_model.to(device)
    return ram_model


def load_tag2text(device: str) -> Any:
    """
    Load the tag2text model with specific configurations.

    Returns:
        Any: Loaded tag2text model.
    """
    # Filter out certain categories
    delete_tag_index = [i for i in range(3012, 3429)]

    tag2text_model = tag2text(
        pretrained=tag2text_checkpoint,
        image_size=384,
        vit="swin_b",
        delete_tag_index=delete_tag_index,
    )
    # Adjust threshold for tag acquisition
    tag2text_model.threshold = 0.64
    tag2text_model.eval()
    tag2text_model = tag2text_model.to(device)
    return tag2text_model


def load_grounding_dino(device: str) -> Any:
    """
    Load the Grounding DINO model with given device settings.

    Returns:
        Any: Loaded Grounding DINO model.
    """
    return load_model(config_file, grounded_checkpoint, device=device)


def load_sam(device: str) -> Any:
    """
    Load the SAM model with given device settings.

    Returns:
        Any: Loaded SAM model.
    """
    return SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
