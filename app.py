
import streamlit as st
import requests
from io import BytesIO
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
import os


st.set_option('deprecation.showfileUploaderEncoding',False)

PATH = r"C:\Users\Rainer\Desktop\trash_streamlit"
logo = Image.open(r"{}\xavi.jpg".format(PATH))

col1, col2 = st.columns(2)

with col1:
  st.header("hi i'm marc the trash classifier")
  st.image(logo.resize((250,250)))

with col2:
  st.title("and i'm hungry for environemntal sustainability")
  st.subheader("feed me an image and i'll try my best to tell you what it is")
  st.write("trained using mobileNet")

st.subheader("Check out our demo video here!")
st.video("https://www.youtube.com/watch?v=Yhq7OZPlP7Y&ab_channel=ClarissaBellaJew", format='video/mp4', start_time=0)

st.subheader("Upload a file for classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model((r'{}\model.h5'.format(PATH)),custom_objects={'KerasLayer':hub.KerasLayer})
  return model

with st.spinner('loading model...'):
  model = load_model()

def decode_img(image):
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    ##test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

def predict(image):
  predictions = model.predict(decode_img(image))
  st.write(predictions)
  result = f"{classes[np.argmax(predictions)]} with a { (100 * np.max(predictions)).round(2) } percent confidence."
  return result


file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

if file_uploaded is not None:    
    image = Image.open(file_uploaded)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis("off")
    st.pyplot(fig)
    st.subheader(predict(image))

st.subheader("orrr classify a randomly generated image here!")

if st.button("surprise me", key="random_classify"):
  directory = r"{}\unseen_test".format(PATH)
  i = np.random.randint(len(os.listdir(directory)))
  choice = Image.open(r"{}\{}".format(directory,os.listdir(directory)[i]))
  fig = plt.figure()
  choice = choice.resize((300,300))
  plt.imshow(choice)
  plt.axis("off")
  st.pyplot(fig)
  st.subheader(predict(choice))
  st.balloons()
