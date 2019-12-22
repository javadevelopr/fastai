#!/usr/bin/env python
# File Name: public_model.py
#
# Date Created: Dec 21,2019
#
# Last Modified: Sat Dec 21 21:48:43 2019
#
# Author: samolof
#
# Description:	
#
##################################################################
from fastai import *
from fastai.vision import *
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os, requests, tempfile

path=Path.cwd()
tempImgPath= path/'tmp'

def download(url):
    try:
        data = requests.get(url)
        imgData = data.content
        
        tmpImgFile = tempfile.TemporaryFile(dir=tempImgPath)
        tmpImgFile.write(imgData)
    

        return (imgData, tmpImgFile)
    except:
        return None

learner = load_learner(path) 

st.title('Popular Electric car models image classifier')

imgURL = st.sidebar.text_input('Enter image url:')

st.subheader('Image')
if imgURL != None and imgURL !='':
    img,imgFile = download(imgURL)
    st.image(img)
    
    fImg = open_image(imgFile)
    pred_class, pred_idx, outputs = learner.predict(fImg)
    st.write('Predicted class:', pred_class)

    imgFile.close()
else:
    st.text('No image yet')


