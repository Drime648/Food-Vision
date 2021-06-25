import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import tensorflow as tf


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def main():
    """Run this function to display the Streamlit app"""
    # st.info(__doc__)
    # st.markdown(STYLE, unsafe_allow_html=True)
 
    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()
 
    if not file:
        show_file.info("Please upload a file of type: " + "/".join(["png", "jpg"]))
        return
 
    content = file.getvalue()

    show_file.image(file)
    file.close()
 
main()


