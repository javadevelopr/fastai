#!/usr/bin/env python
# File Name: public_model.py
#
# Date Created: Dec 21,2019
#
# Last Modified: Mon Dec 23 19:41:57 2019
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



def download(fileName):
    filePath = tmpDir/fileName
    if os.path.exists(filePath):
        if os.path.getsize(filePath) == TRAINING_MODELS[fileName][1]:
            return

    warning, progressBar = None, None
    try:
        warning = st.warning("Downloading %s..." % fileName)
        progressBar = st.progress(0)

        url = TRAINING_MODELS[fileName][0]

        with open(filePath, 'wb') as outfile:
            with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    contentLength = len(r.content)
                    counter = 0.0
                    
                    for data in r.iter_content(chunk_size=8192):
                        if not data: break
                        counter += len(data)
                        outfile.write(data)

                        warning.warning("Downloading %s... (%6.2f/%6.2f MB)" % (fileName, counter, contentLength))
                        progressBar.progress(min(counter/contentLength, 1.0))
    finally:
        if warning is not None:
            warning.empty()
        if progressBar is not None:
            progressBar.empty()




#@st.cache
def downloadImage(url):
    try:
        data = requests.get(url)
        imgData = data.content
        
        tmpImgFile = tempfile.TemporaryFile(dir=tmpDir)
        tmpImgFile.write(imgData)
    

        return (imgData, tmpImgFile)
    except:
        return None


IMAGE_WIDTH=400
path=Path.cwd()
tmpDir= path/'.tmp'
tmpDir.mkdir(exist_ok=True)


TRAINING_MODELS = {
        "car_vs_other.pkl" : ["https://javadevelopr865-fastai.s3-us-west-1.amazonaws.com/car_vs_other.pkl", 87390512],
        "trained_model.pkl": ["https://javadevelopr865-fastai.s3-us-west-1.amazonaws.com/trained_model.pkl",102802036]
}

correct = wrong = 0

def main():


    st.title('Name That (Electric)Car')


    models = list(TRAINING_MODELS.keys())

    with st.spinner('Downloading pre-trained model...'):
        for fileName in models:
            download(fileName)


    imgURL = st.text_input('Enter url for image:')

    imgFile = st.sidebar.file_uploader('Upload image file(s):')
    
    testIfCar = load_learner(path/'.tmp', models[0]) 
    learner = load_learner(path/'.tmp', models[1])

    if imgURL != None and imgURL !='':
        img,imgFile = downloadImage(imgURL)
        st.image(img, width=IMAGE_WIDTH)
    
        tmpImg = open_image(imgFile)
        pred_class, pred_idx, outputs = testIfCar.predict(tmpImg)
        
        if pred_class.__str__() == 'other':
           st.info("Image doesn't appear to contain a car. Try again.")
           return
        else:
            pred_class, _, _ = learner.predict(tmpImg)
            st.write(pred_class)


        option = st.selectbox(
            'Is this correct?',
            ('Yes', 'No')
        )

        global correct, wrong
        if option == 'Yes':
            correct += 1
        elif option == 'No':
            wrong += 1

        #cprogress = st.progress(correct)
        #wprogress = st.progress(wrong)

        imgFile.close()
    else:
        st.text('No image yet')

if __name__=="__main__":
    main()
