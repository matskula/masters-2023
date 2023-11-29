import tensorflow as tf
import keras_cv
from model import class_mapping, model


def detect(
    img,
    path: str
):
    inference_resizing = keras_cv.layers.Resizing(
        640, 640, pad_to_aspect_ratio=True
    )
    img = inference_resizing.call(img)
    img = tf.reshape(img, (1, *img.shape))
    y_pred = model.predict(img)
    y_pred = keras_cv.bounding_box.to_ragged(y_pred)
    keras_cv.visualization.plot_bounding_box_gallery(
        img,
        value_range=(0, 255),
        bounding_box_format='xyxy',
        y_pred=y_pred,
        scale=4,
        rows=1,
        cols=1,
        font_scale=0.7,
        class_mapping=class_mapping,
        path=path
    )
