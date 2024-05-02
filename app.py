import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from gradio_image_prompter import ImagePrompter


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

    # Run inference.
    print('Running inference using ', model_name)
    predicted_logits, predicted_iou = model(
        input_image,
        input_points,
        input_labels,
    )

    # post processing
    mask = tf.greater_equal(predicted_logits[0, 0, 0, :, :], 0).numpy()
    masked_image_np = test_image.copy().astype(np.uint8) * mask[:, :, None]

    return (masked_image_np, points_prompts)


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
                output_image = gr.Image(show_label=False, interactive=False)
                output_points = gr.Dataframe(label="Points")

        prompt_button.click(
            run_model,
            inputs=[image_data, model_choose],
            outputs=[output_image, output_points])

        clear_button.add([image_data, output_image, output_points])
        clear_button.click(lambda: None)

    return sam_interface


if __name__ == "__main__":
    demo = sam_demo()
    demo.queue()
    demo.launch()
