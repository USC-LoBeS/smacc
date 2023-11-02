#!/bin/bash
#$ -S /bin/bash

# Export python environment
export PATH=./anaconda3/envs/smacc/bin:$PATH

SCRIPT_DIR="$(pwd)"
# Input directory of T1
dirI=${SCRIPT_DIR}/input

# Subject list for T1's to be processed
dirS=${SCRIPT_DIR}/subject_list.txt

# Model folder path where the weights for the UNet and AutoQC model is stored
dirM=${SCRIPT_DIR}/model

# Output directory 
dirO=${SCRIPT_DIR}/output
mkdir -p ${dirO}
mkdir -p ${dirO}/segmentation
mkdir -p ${dirO}/metrics

# Modality of the image (T1/T2/FLAIR)
modality=T1

#########################################################################################################################
########## DO NOT CHANGE THIS PATH ########## 
# Python Script path 
script=${SCRIPT_DIR}/scripts
cd ${script}

#########################################################################################################################
# Step 1: Generating CC segmentation------>
python generate_segmentations.py --inp ${dirI} --model_path ${dirM} --out ${dirO} --modality ${modality}

#########################################################################################################################
# Step 2: Extract and Collate CC metrics------->
for entry in "${dirO}/segmentation/nifti"/*
do
	IFS='/'
	read -ra newarr <<< "$entry"
	subj=${newarr[-1]}
	savename=${subj::-7}
	IFS=' '

	python extract_metrics.py --mask ${entry} --output ${dirO}/metrics/${savename}.csv
	echo "Done with ${entry}"
done
python collate_metrics.py --inp ${dirO}/metrics --out ${dirO}

##########################################################################################################################
# Step 3: Auto QC for generated CC segmentations------>
python auto_qc.py --inp ${dirO} --model ${dirM} --out ${dirO}

