#!/bin/bash
#$ -S /bin/bash

# Export python environment
export PATH=./Anaconda3/envs/smacc/bin:$PATH

# Input directory of T1
dirI=./input

# Subject list for T1's to be processed
dirS=./subject_list.txt

# Model folder path where the weights for the UNet and AutoQC model is stored
dirM=./model

# Output directory 
dirO=./output
mkdir -p ${dirO}
mkdir -p ${dirO}/segmentation
mkdir -p ${dirO}/metrics
mkdir -p ${dirO}/QC

# Modality of the image (T1/T2/FLAIR)
modality=T1

#########################################################################################################################
########## DO NOT CHANGE THIS PATH ########## 
# Python Script path 
script=./scripts
cd ${script}

#########################################################################################################################
# Step 1: Generating CC segmentation------>
python generate_segmentations.py --inp ${dirI} --model_path ${dirM} --out ${dirO} --modality ${modality}

#########################################################################################################################
# Step 2: Extract and Collate CC metrics------->
for entry in "${dirO}/segmentation/nifti"/*
do
	python extract_metrics.py --mask ${entry} --output ${dirO}/metrics/${entry}.csv
	echo "Done with ${entry}"
done
python collate_metrics.py --inp ${dirO}/metrics --out ${dirO}

##########################################################################################################################
# Step 3: Auto QC for generated CC segmentations------>
python auto_qc.py --inp ${dirO} --model ${dirM} --out ${dirO}

