import argparse

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from utils.model import load_saved_model, load_tflite


def run_model(model, input_image, input_points, input_labels, use_tflite):
    if use_tflite:
        print("Run inference with TFLite model.")
        # Run inference with TFLite model.
        # Get input and output tensors.
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Set the value of the input tensor
        model.set_tensor(input_details[0]['index'], input_labels)
        model.set_tensor(input_details[1]['index'], input_image)
        model.set_tensor(input_details[2]['index'], input_points)

        # Run the model
        model.invoke()

        # Get the Output data
        predicted_logits = model.get_tensor(output_details[0]['index'])
        predicted_iou = model.get_tensor(output_details[1]['index'])
    else:
        print("Run inference with SavedModel.")
        predicted_logits, predicted_iou = model(
            input_image,
            input_points,
            input_labels,
        )

    return predicted_logits, predicted_iou


def main(use_tflite, model):
    # Load pretrained EfficientSAM-S / EfficientSAM-Ti model
    model_name = "efficient_sam_vitt"
    if model == "small":
        model_name = "efficient_sam_vits"
    saved_model_path = f"weights/saved_model/{model_name}"
    tflite_path = f"weights/{model_name}.fp32.tflite"

    model = load_saved_model(saved_model_path)
    if use_tflite:
        model = load_tflite(tflite_path)

    # Processing the image
    test_image = cv2.imread("images/dogs.jpg")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (1024, 1024))
    input_image = test_image / 255. # 進行圖像歸一處理
    input_image = tf.expand_dims(input_image, axis=0) # (B, H, W, C)
    input_image = tf.cast(input_image, dtype=tf.float32)

    # Points Segmentations
    input_points = tf.constant([[[[500, 630], [580, 630]]]], dtype=tf.float32)
    input_labels = tf.constant([[[1, 1]]], dtype=tf.float32)

    # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
    predicted_logits, predicted_iou = run_model(
    model, input_image, input_points, input_labels, use_tflite)

    mask = tf.greater_equal(predicted_logits[0, 0, 0, :, :], 0).numpy()
    masked_image_np = test_image.copy().astype(np.uint8) * mask[:, :, None]
    Image.fromarray(masked_image_np).save(f"images/dogs_{model_name}_point_mask.png")

    # Generate a image matting with the mask
    # make image that has alpha channel (background transparent)
    #
    # 1. Create an Alpha channel
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    # 2. Set the foreground part in the mask to 255 (completely opaque)
    alpha_channel[mask] = 255
    # 3. Convert the original image to 4 channels (RGBA)
    rgba_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2RGBA)
    # 4. Add the Alpha channel to the image
    rgba_image[:, :, 3] = alpha_channel
    # 5. Save the image with alpha channel
    Image.fromarray(rgba_image).save(f"images/dogs_{model_name}_point_mask_with_alpha.png")

    # Visualize the results of EfficientSAM-Ti
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
    ax1.imshow(test_image)
    ax1.axis("off")
    ax2.imshow(masked_image_np)
    ax2.axis("off")
    ax3.imshow(rgba_image)
    ax3.axis("off")
    fig.tight_layout()
    plt.show()

    # Box Segmentations
    x1 = 385
    y1 = 300
    x2 = 800
    y2 = 1000
    input_points = tf.constant([[[[x1, y1], [x2, y2]]]], dtype=tf.float32)
    input_labels = tf.constant([[[2, 3]]], dtype=tf.float32)

    predicted_logits, predicted_iou = run_model(
        model, input_image, input_points, input_labels, use_tflite)

    mask = tf.greater_equal(predicted_logits[0, 0, 0, :, :], 0).numpy()
    masked_image_np = test_image.copy().astype(np.uint8) * mask[:, :, None]
    Image.fromarray(masked_image_np).save(f"images/dogs_{model_name}_box_mask.png")

    # Generate a image matting with the mask
    # make image that has alpha channel (background transparent)
    #
    # 1. Create an Alpha channel
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    # 2. Set the foreground part in the mask to 255 (completely opaque)
    alpha_channel[mask] = 255
    # 3. Convert the original image to 4 channels (RGBA)
    rgba_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2RGBA)
    # 4. Add the Alpha channel to the image
    rgba_image[:, :, 3] = alpha_channel
    # 5. Save the image with alpha channel
    Image.fromarray(rgba_image).save(f"images/dogs_{model_name}_box_mask_with_alpha.png")

    # Visualize the results of EfficientSAM-Ti
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))
    ax1.imshow(test_image)
    ax1.axis("off")
    ax2.imshow(masked_image_np)
    ax2.axis("off")
    ax3.imshow(rgba_image)
    ax3.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tflite", action="store_true", default=False, help="choose model type")
    parser.add_argument("--model", default='tiny', choices=['tiny', 'small'], help='Choose which model you want to use.')
    args, _ = parser.parse_known_args()
    main(**vars(args))

