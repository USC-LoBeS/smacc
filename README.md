# SMACC-MRI: 
## Segment, Measure and Auto QC the midsagittal Corpus Callosum

This automated pipeline can be used for accurate Corpus Callosum (CC) segmentation across multiple MR modalities (T1, T2 and FLAIR) and extract a variety of features to describe the shape of the CC. We also include an automatic quality control function to detect poor segmentations using Machine Learning.

<p align="center">
<img width="689" alt="workflow" src="https://github.com/USC-LoBeS/smacc/assets/39843804/b9b38025-c391-4da4-8e86-280068447d0b">
</p>

## How to use the tool:
* Clone the github directory using:
	```bash
	git clone https://github.com/USC-LoBeS/smacc.git
	```
 
## Virtual environment:
* Navigate to the "smacc" folder and then create a virtual environment using the smacc_env.yml file:
	* For Linux users:
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
* Add modality of the images to be processed (T1/T2/FLAIR) on line 24.

After the changes are made to the .sh file, run this tool using:
```bash
./run_CC.sh
```
The final output will be "metrics_Final.csv" in the output folder which will have all the metrics and a column "QC label" indicating whether the segmentations were accurate(0)/fail(1).


## Demo:
* To run this tool with the given examples (T1's) in the input folder:
	* Get the python path for the virtual environment using the following commands in the terminal:
	```bash
	conda activate smacc
	which python
	conda deactivate
	```
   	* The path would look something like this:
	```bash
	/Users/User123/anaconda3/envs/smacc/bin
	```
 	 * Add this python path to the run_CC.sh file and run the .sh file using:
  	```bash
	./run_CC.sh
	```

#### If you use this code, please cite the following paper:
##### Shruti P. Gadewar, Elnaz Nourollahimoghadam, Ravi R. Bhatt, Abhinaav Ramesh, Shayan Javid, Iyad Ba Gari, Alyssa H. Zhu, Sophia Thomopoulos, Paul M. Thompson, and Neda Jahanshad. "A Comprehensive Corpus Callosum Segmentation Tool for Detecting Callosal Abnormalities and Genetic Associations from Multi Contrast MRIs." ArXiv (2023): arXiv-2305. (https://arxiv.org/abs/2305.01107)
