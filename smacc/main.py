#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 2024

@author: shruti
"""

import os
import sys
import argparse
from smacc.utils.generate_segmentations import get_segmentation
from smacc.utils.extract_metrics import get_shape_metrics
from smacc.utils.qc import auto_qc

import warnings
warnings.simplefilter("ignore")

def run_smacc():
    print("""                                 ...'''...                                   
                            .;cdk0KNNNNNXK0ko;..                             
                         'lOXWMMMMMMMMMMMMMMMWX0xc;'.                        
                      .:kNMMMMMMMWNNX00KXNNNWWMMMMWNKOxoc;'.                 
                  .'cxXWMMMMWXOdc;''......'',:c:lodkKNWMMWNK0Oxdddddl;.      
            ..,:ox0NMMMMWXko;.                      .':oONMMMMMMMMMMMWKo.    
       .;lx0KNWMMMMMWXko;.                               'lONMMMMMMMMMMW0,   
     'dKWMMMMMMMMWKx:.                                      'lONMMMMMMMMMK;  
    cXMMMMMMMMMWk;.    SSSS   MM   MM     A      CCCC   CCCC  ;kWMMMMMMMM0' 
   .kMMMMMMMMMMx.     SS      M M M M    A A    C      C       .kMMMMMMMMX; 
   .dWMMMMMMMMM0:.     SSSS   M  M  M   A   A   C      C        oWMMMMMMXc  
    .xNMMMMMMMMMWO:.      SS  M     M  A A A A  C      C        ,0MMMMMKc   
      ;d0NMMMMMMMMWO:  SSSS   M     M  A     A   CCCC   CCCC     .cddoc.    
        .':loodxkxoo;                                                      

""")
    print("\n########################", flush=True)
    print("If you are using smacc, please cite the following paper:", flush=True)
    print("Gadewar SP, Nourollahimoghadam E, Bhatt RR, Ramesh A, Javid S,"
          "Gari IB, Zhu AH, Thomopoulos S, Thompson PM, Jahanshad N. "
          "A Comprehensive Corpus Callosum Segmentation Tool for Detecting "
          "Callosal Abnormalities and Genetic Associations from Multi Contrast MRIs. "
          "Annu Int Conf IEEE Eng Med Biol Soc. 2023 Jul;2023:1-4. "
          "doi: 10.1109/EMBC40787.2023.10340442. PMID: 38083493.", flush=True)
    print("########################\n", flush=True)

    # Argument Parser
    parser = argparse.ArgumentParser(description="Segmentation, Metrics Extraction and Auto QC the Corpus Callosum")
    parser.add_argument("-f", "--filename", type=str, help="List of filenames (in .txt format) with absolute path to the niftis and the output filenames", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output directory", required=True)
    parser.add_argument("-m", "--modality", type=str, help="T1/T2/FLAIR modality", required=True)
    parser.add_argument("-q", "--QC", action='store_true', help="Perform AutoQC on segmentations")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    filenames = args.filename
    out = args.output
    modality = args.modality
    qc_flag = args.QC
     
    # Create output directory
    os.makedirs(out, exist_ok=True)
    
    # Read subjectList with absolute paths from a text file into a list
    file = open(filenames, "r")
    filenameList = [line.strip() for line in file]
    file.close()
    
    print("Generating CC segmentations and extracting shape metrics-->")
    for ls in filenameList:  
        filePath = ls.split(" ")[0]
        savename = ls.split(" ")[1]
        
        # Generate segmentations
        get_segmentation(
            filePath, 
            out, 
            savename,
            modality)
        
        # Extract shape metrics
        get_shape_metrics(
            os.path.join(out, "segmentation", "nifti"),
            savename,
            os.path.join(out, "metrics"))
    
    # AutoQC for generated CC segmentations using the shape metrics
    print(qc_flag)
    auto_qc(
        os.path.join(out, "metrics"),  
        out,
        qc_flag)
    
    print("Finished!")


if __name__ == '__main__':
    run_smacc()
    
