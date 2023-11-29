from uuid import uuid4

import streamlit as st
import tensorflow as tf

from predictor import detect


st.title('Military aircraft detection')
uploaded_file = st.file_uploader('Upload satellite military airfield image')
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = tf.image.decode_jpeg(bytes_data, channels=3)
    tmp_id = uuid4()
    path = f'images_output/{tmp_id}.jpg'
    detect(image, path)
    st.image(path, use_column_width='always', caption='Detection result')
