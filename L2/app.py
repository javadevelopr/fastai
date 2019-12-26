#!/usr/bin/env python
# File Name: public_model.py
#
# Date Created: Dec 21,2019
#
# Last Modified: Wed Dec 25 21:03:25 2019
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
import os, requests, tempfile
import urllib, PIL

IMAGE_WIDTH=400
path=Path.cwd()
tmpDir= path/'.tmp'
tmpDir.mkdir(exist_ok=True)

AWS_PREFIX="https://javadevelopr865-fastai.s3-us-west-1.amazonaws.com"


TRAINING_MODELS = {
        "pre_train.pkl" : 102787490,
        "trained_model.pkl": 102803581
}

car_class = {
        'other' : 'ICE clunker 😷 🏭, model unknown',
        'model_x' : 'Tesla Model X',
        'model_s' : 'Tesla Model S',
        'model_3' : 'Tesla Model 3',
        'taycan'  : 'Porsche Taycan',
        'ff91'    : 'Faraday Future FF91',
        'volt'    : 'Chevy Volt',
        'bolt'   :  'Chevy Bolt',
        'lucidair' : 'Lucid Air',
        'fisker'  : 'Fisker Karma',
        'rimac' : 'Rimac Concept 1/2'
}

WRONG_PNG = 'wrong.png'
CORRECT_PNG= 'green_check.png'

intro_text= """
        Upload an image of a car or a url of an image of car.
I will try to guess if it's one of 9 popular electric car models.
Image should contain only one car so I can give it my best shot.

You can find images on [Bing](https://www.bing.com/image) or [Google Image search](https://images.google.com).
Make sure the url links to an actual image (E.g on google image search, right click on image and choose 'Copy Link Location')
"""





@st.cache
def _pilImg(img):
    return PIL.Image.open(img)


def download(fileName):
    filePath = tmpDir/fileName
    if os.path.exists(filePath):
        if os.path.getsize(filePath) == TRAINING_MODELS[fileName]:
            return

    warning, progressBar = None, None
    try:
        warning = st.warning("Downloading %s..." % fileName)
        progressBar = st.progress(0)

        url = AWS_PREFIX + "/" + fileName

        with open(filePath, 'wb') as outfile:
            
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter =0.0
                while True:
                    data = response.read(8192)
                    if not data: break
                    
                    counter += len(data)
                    outfile.write(data)

                    warning.warning("Downloading %s... (%6.2f/%6.2f Bytes)" % (fileName, counter, length))
                    progressBar.progress(min(counter/length, 1.0))
    finally:
        if warning is not None:
            warning.empty()
        if progressBar is not None:
            progressBar.empty()



def compositeImage(fgImage, bgImage):

    bw,bh = bgImage.size
    fw,fh = fgImage.size

    fgImage = fgImage.resize((min(bw,fw//2), min(bh,fh//2)))
    offset = ( (bw - fgImage.size[0]) //2, (bh - fgImage.size[1]) // 2)
    #offset=(0,0)
    tmpImage = bgImage.copy()
    tmpImage.paste(fgImage, offset, mask=fgImage)

    return tmpImage


#@st.cache
def downloadImage(url):

    def _imageURLFromURL(url):
        """Extract the image url from the url for Bing and Google image search"""
        from urllib.parse import unquote
        try:
            query = url.split('&')
            a=list(filter(lambda x: len(x) ==2, map(lambda x: x.split('='), query)))
            return list(map(lambda x: urllib.parse.unquote(x[1]), filter(lambda x: 'imgurl' in x[0] or 'mediaurl' in x[0],a)))[0]
        except:
            return url
        return url


    try:
        data = requests.get(_imageURLFromURL(url))
        imgData = data.content
        

        return io.BytesIO(imgData)
    except:
        return None

import io    
def predict_image(img):

    img = open_image(img)

    models = list(TRAINING_MODELS.keys())
    prelearner = load_learner(path/'.tmp', models[0]) 
    learner = load_learner(path/'.tmp', models[1])

    pred_class, pred_idx, outputs = prelearner.predict(img)
    
    if pred_class.__str__() == 'other':
       return None
    else:
        pred_class, _, _ = learner.predict(img)
        return pred_class



def main():

    st.title('Tag That (Electric)Car')

    img = imgFromFile = imgURL = option =  None

    models = list(TRAINING_MODELS.keys())

    with st.spinner('Downloading training models ...'):
        for fileName in models:
            download(fileName)


    st.markdown(intro_text)

    imgURL = st.text_input('Enter url for image:')

    if imgURL != None and imgURL !='':
        img = downloadImage(imgURL)

   
    else:
        img = None
        imgFromFile = st.file_uploader('Upload image file:')

        if imgFromFile != None:
            img = imgFromFile
       
        else:
            st.text('No image yet')
            return

    try:
        if option is not None: option.empty()
        imgFrame=st.image(img,  use_column_width=True)
        cl = predict_image(img)

        if cl is not None:
            st.success("I see a %s" % car_class[cl.__str__()])
        else:
            st.info("The image doesn't seem to be an image of car. Try again?")
        
        option = st.selectbox(
            'Is this correct?',
            ('Yes', 'No'), index = 0
        )

        if option == 'Yes':
            pass
            #imgFrame.image(compositeImage(_pilImg(CORRECT_PNG), _pilImg(img)), use_column_width=True)
        elif option == 'No':
            imgFrame.image(compositeImage(_pilImg(WRONG_PNG), _pilImg(img)), use_column_width=True)  
    except:
        st.error("Unable to load image. Are you sure this is an image file/URL?")
        return

if __name__=="__main__":
    main()
