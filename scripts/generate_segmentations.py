import warnings
warnings.simplefilter("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import random
import tensorflow as tf
import argparse

import nibabel as nib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dropout, Lambda, MaxPooling2D,Conv2D, Conv2DTranspose,Input
from tensorflow.keras.metrics import MeanIoU,Precision,Recall
from tqdm import tqdm

tf.config.threading.set_inter_op_parallelism_threads(1)

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--inp", help=""" Input folder path """)
parser.add_argument("--model_path", help="""Path to where the weights for the model are saved """)
parser.add_argument("--out", help="""output folder path """)
parser.add_argument("--modality", help="""T1 or T2 or FLAIR""")
args = parser.parse_args()

dest_dir=args.out

if args.modality=="FLAIR" or args.modality=="Flair" or args.modality=="flair":
    weight_path=args.model_path+"/FLAIROnly_Final.h5"
elif args.modality=="T2" or args.modality=="t2" :
    weight_path=args.model_path+"/T2Only_Final.h5"
else:
    weight_path=args.model_path+"/T1Only_Final.h5"

img_dir=args.inp
print(weight_path)


def getUnet():
    inputs = Input((imgh, imgw, N_CHANNELS))
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
    outputs = Conv2D(NUM_CLASSES, (1, 1), activation='softmax') (c9)
  
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy',Precision(),Recall(),MeanIoU(num_classes=NUM_CLASSES)])
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
    img1 = np.reshape(img1,(1,imgh, imgw, N_CHANNELS))
    return img1,affine,header,ind

def process_op(img,path,affine,header,dest_pathl):
    predt = (img > 0.5).astype(np.uint8)
    pred = reverse_one_hot(img)
    pred_vis_image = colour_code_segmentation(pred, label_values)
    predt = reverse_one_hot(predt)
    res=np.reshape(np.squeeze(pred_vis_image),(256,256))
    cv2.imwrite(dest_pathl,res[20:-18,38:-36])
    res1 = np.zeros((182, 218, 182))
    res1[89,1:,1:] = res[20:-19,38:-37]
    img = nib.Nifti1Image(res1, affine=affine, header=header, extra=None, file_map=None)
    nib.save(img, path)
    
label_values=[0,255]
imgh, imgw, N_CHANNELS,NUM_CLASSES=256,256,1,2
def check_image_norm(img):
    if (np.max(img)>256) or (np.min(img)<0):
        return False
    else:
        return True

print("Running Predictions.....")
fol=""
model=getUnet()
model.load_weights(weight_path)


path=img_dir+fol+"/"
print(path)
fol="segmentation"
dest_path=dest_dir+"/"+fol+"/nifti/"
dest_pathi=dest_dir+"/"+fol+"/Images_PNG/"
dest_pathl=dest_dir+"/"+fol+"/Labels_PNG/"
os.makedirs(dest_path, exist_ok=True)
os.makedirs(dest_pathi,exist_ok=True)
os.makedirs(dest_pathl,exist_ok=True)

for it in tqdm(os.listdir(path)):
    try:
        if it[-7:]==".nii.gz":	
            img,affine,header,ind=load_image(path+it,dest_pathi+it[:-7]+".png")
            if check_image_norm(img):
                process_op(model.predict(img),dest_path+it,affine,header,dest_pathl+it[:-7]+".png")
            else:
                print(" Normalizing - ",it) 
                print("Preprocessing Done!")
                process_op(model.predict(img),dest_path+it,affine,header,dest_pathl+it[:-7]+".png")

    except:
        print("Failed--",it)
        pass
        
