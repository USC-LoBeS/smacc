# SMACC-MRI: 
## Segment, Measure and AutoQC the midsagittal Corpus Callosum

This automated pipeline can be used for accurate Corpus Callosum (CC) segmentation across multiple MR modalities (T1, T2 and FLAIR) and extract a variety of features to describe the shape of the CC. We also include an automatic quality control function to detect poor segmentations using Machine Learning.

<p align="center">
<img width="689" alt="workflow" src="https://github.com/ShrutiGadewar/smacc/assets/39843804/b9b38025-c391-4da4-8e86-280068447d0b">
</p>

## How to use the tool:
Clone the github directory using:
```bash
git clone https://github.com/ShrutiGadewar/smacc.git
```
 
## Virtual environment:
Navigate to the "smacc" folder and then create a virtual environment using the requirements.txt file:
```bash
conda create -n smacc python==3.11 -y
conda activate smacc
pip install -r requirements.txt
pip install .
```

## Input Preprocessing:
All the MR images should be registred to MNI 1mm template(182 X 218 X 182) with 6dof. You can use the template provided in the "model" folder on github. You can use the FSL's flirt command for linear registration:
```bash
flirt -in ${inpdir}/${subj}.nii.gz \
	-ref ${MNI_1mm_template} \
  	-out ${outdir}/${subj} \
 	-dof 6 \
  	-cost mutualinfo \
  	-omat ${outdir}/matrices/${subj}_MNI_6p.xfm
```

## Test the tool:
```bash
smacc -f ./subject_list.txt -o ./smacc_output -m t1
```
-f : Text file with a list of absolute paths to the niftis to be processed and names to save the outputs for each subject. Check example text file "subject_list.txt" provided. <br />
-o : Absolute path of output folder <br />
-m : Modality of the images to be processed (t1/t2/flair) <br />
-q : Optional flag to perform Automated QC on the segmentations <br />
The final output is a csv which will contain all the extracted shape metrics and a column "QC label" indicating whether the segmentations were accurate(0)/fail(1) if the QC flag is provided.


## If you use this code, please cite the following paper:
Gadewar SP, Nourollahimoghadam E, Bhatt RR, Ramesh A, Javid S, Gari IB, Zhu AH, Thomopoulos S, Thompson PM, Jahanshad N. A Comprehensive Corpus Callosum Segmentation Tool for Detecting Callosal Abnormalities and Genetic Associations from Multi Contrast MRIs. Annu Int Conf IEEE Eng Med Biol Soc. 2023 Jul;2023:1-4. doi: 10.1109/EMBC40787.2023.10340442. PMID: 38083493.

