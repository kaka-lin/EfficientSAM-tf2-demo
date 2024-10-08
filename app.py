import os

import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import gradio as gr
from gradio_image_prompter import ImagePrompter

from utils.ui_components import ToolButton


def save_files(images):
    fullfns = []

    # make output dir
    saved_path = "log/images/"
    os.makedirs(saved_path, exist_ok=True)

    # save image for download
    output_image = saved_path + 'output.png'
    Image.fromarray(images).save(output_image)
    fullfns.append(output_image)

    return gr.File(value=fullfns, visible=True)


def load_model():
    """Load pretrained EfficientSAM-Ti & EfficientSAM-S model"""
    models = {}
    weight_names = ["efficient_sam_vitt", "efficient_sam_vits"]
    model_names = ["EfficientSAM-Ti", "EfficientSAM-S"]
    for weight_name, model_name in zip(weight_names, model_names):
        saved_model_path = f"weights/saved_model/{weight_name}"
        models[model_name] = tf.keras.models.load_model(saved_model_path)
    return models


models = load_model()


def process_points(points_prompts, scale_w, scale_h):
    # input_points: [batch_size, num_queries, max_num_pts, 2], 2: x, y
    # input_labels: [batch_size, num_queries, max_num_pts], 1: point, 2: box topleft, 3: box bottomright, 4: None
    input_points = []
    input_labels = []
    # processing the points to point and label
    for points in points_prompts:
        if points[2] == 1.0:
            input_points.append([points[0], points[1]])
            input_labels.extend([points[2]])
        elif points[2] == 2.0 and points[5] == 3.0:
            input_points.append([points[0], points[1]])
            input_points.append([points[3], points[4]])
            input_labels.extend([points[2], points[5]])

    # rescale to 1024x1024 based on the original image size
    input_points = np.around(np.array(input_points) * [scale_w, scale_h])
    input_points = tf.reshape(tf.constant(input_points, dtype=tf.float32), [1, 1, -1, 2])
    input_labels = tf.reshape(tf.constant(input_labels, dtype=tf.float32), [1, 1, -1])

    return input_points, input_labels


def run_model(prompts, model_name):
    # Choose model
    model = models[model_name]

    # Load image and points
    test_image = prompts["image"]
    points_prompts = prompts["points"]

    # processing the image
    scale_h, scale_w = 1024 / test_image.shape[0], 1024 / test_image.shape[1]
    test_image = cv2.resize(test_image, (1024, 1024))
    input_image = test_image / 255. # 進行圖像歸一處理
    input_image = tf.expand_dims(input_image, axis=0) # (B, H, W, C)
    input_image = tf.cast(input_image, dtype=tf.float32)

    # processing the `points_prompts` to point and label
    input_points, input_labels = process_points(points_prompts, scale_w, scale_h)

    # Run inference.
    print('Running inference using ', model_name)
    predicted_logits, predicted_iou = model(
        input_image,
        input_points,
        input_labels,
    )

    # post processing
    mask = tf.greater_equal(predicted_logits[0, 0, 0, :, :], 0).numpy()
    # masked_image_np = test_image.copy().astype(np.uint8) * mask[:, :, None]

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

    return (rgba_image, points_prompts)


def sam_demo():
    with gr.Blocks() as sam_interface:
        gr.Markdown("EfficientSAM Demo")

        with gr.Row():
            with gr.Column():
                model_choose = gr.Dropdown(value="EfficientSAM-Ti", choices=["EfficientSAM-Ti", "EfficientSAM-S"], label="EfficientSAM Model")
                image_data = ImagePrompter(show_label=False)
                with gr.Row():
                    clear_button = gr.ClearButton()
                    prompt_button = gr.Button("Submit")

            with gr.Column():
                output_image = gr.Image(show_label=False, interactive=False, image_mode='RGBA')
                output_points = gr.Dataframe(label="Points")

                save_button = ToolButton('💾', elem_id=f'save_sam')
                download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_sam')

        prompt_button.click(
            run_model,
            inputs=[image_data, model_choose],
            outputs=[output_image, output_points])

        save_button.click(
            fn=save_files,
            inputs=[output_image],
            outputs=[download_files],
        )

        clear_button.add([image_data, output_image, output_points])
        clear_button.click(lambda: None)

    return sam_interface


if __name__ == "__main__":
    demo = sam_demo()
    demo.queue()
    demo.launch()
