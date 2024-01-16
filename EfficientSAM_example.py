import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def load_tflite(model_path):
    """ Run inference with TFLite model. """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    return interpreter


def main():
    # Load pretrained EfficientSAM-Ti model
    model_name = "efficient_sam_vitt"
    tflite_path = "weights/efficient_sam_vitt.fp32.tflite"
    model = load_tflite(tflite_path)

    # Run inference for both EfficientSAM-Ti and EfficientSAM-S models.
    test_image = cv2.imread("images/dogs.jpg")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (1024, 1024))
    input_image = test_image / 255. # 進行圖像歸一處理
    input_image = tf.expand_dims(input_image, axis=0) # (B, H, W, C)
    input_image = tf.cast(input_image, dtype=tf.float32)

    input_points = tf.constant([[[[500, 630], [580, 630]]]], dtype=tf.float32)
    input_labels = tf.constant([[[1, 1]]], dtype=tf.float32)

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

    mask = tf.greater_equal(predicted_logits[0, 0, 0, :, :], 0).numpy()
    masked_image_np = test_image.copy().astype(np.uint8) * mask[:, :, None]
    Image.fromarray(masked_image_np).save(f"images/dogs_{model_name}_mask.png")

    # Visualize the results of EfficientSAM-Ti
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(test_image)
    ax1.axis("off")
    ax2.imshow(masked_image_np)
    ax2.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
