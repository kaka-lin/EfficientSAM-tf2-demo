import tensorflow as tf


def load_saved_model(model_path):
    """ Load the saved model. """
    # model = tf.saved_model.load(model_path)
    model = tf.keras.models.load_model(model_path)
    return model


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
