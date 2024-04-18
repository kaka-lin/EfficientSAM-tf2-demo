import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from gradio_image_prompter import ImagePrompter


demo = gr.Blocks()
# Load pretrained EfficientSAM-Ti model
model_name = "efficient_sam_vitt"
saved_model_path = "weights/saved_model"
model = tf.keras.models.load_model(saved_model_path)


def run_model(prompts):
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
    input_points = np.array(input_points) * [scale_w, scale_h]
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


with demo:
    gr.Markdown("EfficientSAM Demo")

    with gr.Row():
        with gr.Column():
            image_file = ImagePrompter(show_label=False)
            with gr.Row():
                clear_button = gr.ClearButton()
                prompt_button = gr.Button("Submit")

        with gr.Column():
            output_image = gr.Image(show_label=False, interactive=False)
            output_points = gr.Dataframe(label="Points")

    prompt_button.click(
        run_model,
        inputs=image_file,
        outputs=[output_image, output_points])

    clear_button.add([image_file, output_image, output_points])
    clear_button.click(lambda: None)


if __name__ == "__main__":
    demo.launch()
