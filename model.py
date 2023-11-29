import tensorflow as tf
import keras_cv


class_mapping = {
    0: 'Plane'
}

model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.005, momentum=0.9, global_clipnorm=10.0
    ),
    metrics=None,
)
model.load_weights('checkpoint/checkpoint')
model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xyxy",
    from_logits=True,
    iou_threshold=0.2,
    confidence_threshold=0.5,
)