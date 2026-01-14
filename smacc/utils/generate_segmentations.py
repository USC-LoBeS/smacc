#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 2024

@author: shruti
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import importlib_resources

import numpy as np
import cv2
import nibabel as nib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dropout, Lambda, MaxPooling2D, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
tf.config.threading.set_inter_op_parallelism_threads(1)

import warnings
warnings.simplefilter("ignore")



def get_unet():
    inputs = Input((256, 256, 1))
    s = Lambda(lambda x: x / 255) (inputs)
    # Constructive Path / Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    
    # Expansive Path / Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
   
    # Output Layer
    outputs = Conv2D(2, (1, 1), activation='softmax') (c9)
  
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy',Precision(),Recall(),MeanIoU(num_classes=2)])
    model.summary()
    
    return model

def reverse_one_hot(image):
	x = np.argmax(image, axis = -1)
	return x

def colour_code_segmentation(image, label_values):
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]
	return x

def load_image(path,dest_pathi):
    img = nib.load(path)
    affine=img.affine
    header=img.header
    data = img.get_fdata()
    ind = 89
    slice_data = data[ind,:,:]
    print(np.min(slice_data),np.max(slice_data),slice_data.shape)
    img1 = cv2.normalize(slice_data, None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    img1 = img1.astype('uint8')       
    img1 = cv2.equalizeHist(img1)
    cv2.imwrite(dest_pathi,img1)
    print(img1.shape)
    img1=cv2.copyMakeBorder(img1, 19, 19, 37, 37, cv2.BORDER_CONSTANT, (0,0,0))
    print(img1.shape)
    img1 = np.reshape(img1,(1, 256, 256, 1))
    return img1,affine,header,ind

def process_op(img,path,affine,header,dest_pathl):
    predt = (img > 0.5).astype(np.uint8)
    pred = reverse_one_hot(img)
    pred_vis_image = colour_code_segmentation(pred, [0,255])
    predt = reverse_one_hot(predt)
    res=np.reshape(np.squeeze(pred_vis_image),(256,256))
    cv2.imwrite(dest_pathl,res[20:-18,38:-36])
    res1 = np.zeros((182, 218, 182))
    res1[89,1:,1:] = res[20:-19,38:-37]
    img = nib.Nifti1Image(res1, affine=affine, header=header, extra=None, file_map=None)
    nib.save(img, path)
    
def get_segmentation(inp, out, subject, modality):
    # Load models according to the modality
    if modality.upper()=="FLAIR":
    	weight_path = importlib_resources.files('smacc') / 'model/FLAIROnly_Final.h5'
        # weight_path=os.path.join(model_path,"FLAIROnly_Final.h5")
    elif modality.upper()=="T2":
    	weight_path = importlib_resources.files('smacc') / 'model/T2Only_Final.h5'
        # weight_path=os.path.join(model_path,"T2Only_Final.h5")
    elif modality.upper()=="T1":
    	weight_path = importlib_resources.files('smacc') / 'model/T1Only_Final.h5'
        # weight_path=os.path.join(model_path,"T1Only_Final.h5")
    else:
        print("Check modality!")
    print("Weights: ", weight_path)
    
    # Set paths for saving labels, images and nifti outputs
    fol = "segmentation"
    dest_path = os.path.join(out, fol, "nifti")
    dest_pathi = os.path.join(out, fol, "Images_PNG")
    dest_pathl = os.path.join(out, fol, "Labels_PNG")
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(dest_pathi,exist_ok=True)
    os.makedirs(dest_pathl,exist_ok=True)
    
    # Run predictions
    print("Running Predictions.....")
    model=get_unet()
    model.load_weights(weight_path)
    
    try:
        img, affine, header, ind = load_image(inp, os.path.join(dest_pathi,subject+".png"))
        process_op(model.predict(img), os.path.join(dest_path,subject+".nii.gz"), affine, header, os.path.join(dest_pathl,subject+".png"))
    
    except Exception as e:
        print("Failed--", subject)
        print(e)
        pass




