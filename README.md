# SMACC-MRI: 
## Segmentation, Metrics extraction, Automatic quality assessment of the midsagittal Corpus Callosum

This automated pipeline can be used for accurate Corpus Callosum (CC) segmentation across multiple MR modalities (T1, T2 and FLAIR) and extract a variety of features to describe the shape of the CC. We also include an automatic quality control function to detect poor segmentations using Machine Learning.

## How to use the tool:
* Clone the github directory using: git clone https://github.com/USC-LoBeS/smacc.git

## Virtual environment:
* Create a virtual environment using the requirments.txt file:
	* For pip users:
	```bash
	pip install -r requirements.txt
	```

	* For conda users:
	```bash
	conda create --name <env_name> --file requirements.txt
	```

* If you want to create a virtual environment using .yml file:
	```bash
	conda env create -f smacc_env.yml
	```

## Input Preprocessing:
* All the MR images should be registred to MNI 1mm template(182 X 218 X 182) with 6dof. You can use the template provided in the "model" folder on github. You can use the FSL's flirt command for linear registration:
	```bash
	flirt -in ${inpdir}/${subj}.nii.gz \
	 	-ref ${MNI_1mm_template} \
	  	-out ${outdir}/${subj}_MNI_6p \
	  	-dof 6 \
	  	-cost mutualinfo \
	  	-omat ${outdir}/matrices/${subj}_MNI_6p.xfm
	```

## Test the tool:
In run_CC.sh file:
* Once the virtual environment is installed, add the python path on line 5.
* Put all the registred MR images in one folder and put the path for the same on line 8.
* Create a text file with all the subject id's of the MR scans and put the path to the text file on line 11.
* Add the model directory on line 14.
* Set the output path folder where all the results would be generated on line 17.
* Add modality of the images to be processed (T1/T2/FLAIR) on line 23
The final output will be "metrics_Final.csv" in the output folder which will have all the metrics and a column "QC label" indicating whether the segmentations were accurate(0)/fail(1).


#### If you use this code, please cite the following paper:
##### Shruti P. Gadewar, Elnaz Nourollahimoghadam, Ravi R. Bhatt, Abhinaav Ramesh, Shayan Javid, Iyad Ba Gari, Alyssa H. Zhu, Sophia Thomopoulos, Paul M. Thompson, and Neda Jahanshad. "A Comprehensive Corpus Callosum Segmentation Tool for Detecting Callosal Abnormalities and Genetic Associations from Multi Contrast MRIs." ArXiv (2023): arXiv-2305. (https://arxiv.org/abs/2305.01107)