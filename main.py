from tensorflow import keras
import streamlit as st
import numpy as np
from PIL import Image
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras import models, layers



def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((50,50))
    img = np.expand_dims(img, axis=0)
    return img

def limage(image_file):
    img = Image.open(image_file)
    return img

def create_model():
    model = models.Sequential()

    model.add(EfficientNetB0(include_top=False, weights='imagenet',
                             input_shape=(50, 50, 3)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(5, activation="softmax"))

    model.compile(optimizer=Adam(lr=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    return model




st.header('Deep Learning Lab-2 : Cassava Leaf Disease Prediction Challenge Kaggle using CNN')
st.subheader("Upload Image to find the disease in the plant")
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
if image_file is not None:
    file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
    st.write(file_details)
    image = load_image(image_file)
    st.image(limage(image_file), width=250)
    model = create_model()
    model.load_weights('best_baseline_model.h5')
    print("Succesfully loaded weights!!")
    res = np.argmax(model.predict(image))
    name = {"0": "Cassava Bacterial Blight (CBB)", "1": "Cassava Brown Streak Disease (CBSD)","2": "Cassava Green Mottle (CGM)", "3": "Cassava Mosaic Disease (CMD)", "4": "Healthy"}
    st.subheader("Result:"+name[str(res)])
