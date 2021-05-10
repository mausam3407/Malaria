from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
def do_salience(image, model, label):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB) 
    img = cv.resize(img, (150, 150)) / 255.0

    img1 = np.expand_dims(img, axis=0)
    num_classes = 2

    expected_output = tf.one_hot([label] * img1.shape[0], num_classes)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img1, tf.float32)

        tape.watch(inputs)

        predictions = model(inputs)

        loss = tf.keras.losses.categorical_crossentropy(
            expected_output, predictions
        )
    gradients = tape.gradient(loss, inputs)
    grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)
    normalized_tensor = tf.cast(
        255
        * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
        / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
        tf.uint8,)

    normalized_tensor = tf.squeeze(normalized_tensor)
    gradient_color = cv.applyColorMap(normalized_tensor.numpy(), cv.COLORMAP_HOT)
    gradient_color = gradient_color / 255.0
    super_imposed = cv.addWeighted(img, 0.5, gradient_color, 0.5, 0.0)
    return super_imposed

model = tf.keras.models.load_model('malaria_v2.h5')
st.title('Malaria Detection')
st.write('Upload the cell image to check whether the cell is infected or not')

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image
def predict(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    img = cv.resize(img, (150, 150)) / 255.0

    img1 = np.expand_dims(img, axis=0)
    pred = model.predict(img1)
    res = np.argmax(pred)
    acc = np.max(pred)
    return res,acc

uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

col1, col2 = st.beta_columns([2,2])

if uploadFile is not None:
    img = load_image(uploadFile)
    col1.subheader("Original Image")
    col1.image(img,use_column_width='always')
    result,accuracy = predict(img)
    if result==0:
        text = f"Infected, Confidence: {accuracy*100} %"
    else:
        text = f"Un-Infected, Confidence: {accuracy*100} %"
    super1 = do_salience(img,model,result)
    col2.subheader('Result')
    col2.image(super1,caption=text,use_column_width='always')
    st.write('Part coloured with dark Purple/Whitish dots in result image shows the hotspots/ROI of cell')
    st.write("For codes click on this link: https://malaria-detector-with-salience.herokuapp.com/")
else:
    col1.write("Make sure you image is in JPG/PNG Format.") 
    col2.write("Waiting for image")  
