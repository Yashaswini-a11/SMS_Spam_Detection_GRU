import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.set_page_config(
    page_title="SMS Spam Detection",
    layout="centered"
)

st.title("📩 SMS Spam Detection using GRU")

# Load Model
model = load_model("sms_spam_gru.h5")

# Load Tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LEN = 100

st.write(
    "Enter a message below and check whether it is Spam or Ham."
)

message = st.text_area(
    "Enter SMS Message"
)

if st.button("Predict"):

    if message.strip() == "":

        st.warning(
            "Please enter a message."
        )

    else:

        sequence = tokenizer.texts_to_sequences(
            [message]
        )

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN
        )

        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        if prediction > 0.5:

            st.error("🚨 Spam Message")

        else:

            st.success("✅ Ham Message")

        st.metric(
            "Spam Probability",
            f"{prediction*100:.2f}%"
        )