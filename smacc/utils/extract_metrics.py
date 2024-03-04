#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 2024

@author: shruti
"""

import os
import math
import numpy as np
import pandas as pd
import nibabel as nib

import cv2
import skimage.measure
from skimage.morphology import thin
from skimage.transform import rotate
from skimage.measure import regionprops, label, find_contours

import warnings
warnings.simplefilter("ignore")


def extract_cc(image):
    hdr = nib.load(image)
    img_data = hdr.get_fdata()
    midslice = img_data[89,:,:]
    cc = np.rollaxis(midslice, 1)
    return cc


def euclidean_dist(x1, x2, y1, y2):
    try:
        return (np.sqrt(np.sum([np.square(x1-x2), np.square(y1-y2)])))
    except:
        return np.nan


def calc_angle(x1,x2,x3,y1,y2,y3):
    try:
        s1 = (y1-y2)/(x1-x2)
        s2 = (y3-y1)/(x3-x1)
        angle = math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
        return angle
    except:
        return np.nan


def calc_arc_length(a,b):
    try:
        arc_length = 0.5 * math.pi * (3*(a+b) - math.sqrt((3*a + b) * (a + 3*b)))
        return arc_length
    except:
        return np.nan


def thickness_calc(corcal):
    try:
        contours = find_contours(corcal, 0)
        cl = 0
        for i, c in enumerate(contours):
            if len(c) > cl:
                print()
                cl = len(c)
                outline = contours[i]

        dic = {}
        for i in range(0, len(outline)):
            if outline[i][1] in dic.keys():
                dic[outline[i][1]].append(outline[i][0])
            else:
                dic[outline[i][1]] = [outline[i][0]]

        points = []
        for k,v in dic.items():
            if len(v)==2:
                points.append(k)

        thickness=[]
        for point in points:
            thickness.append(np.abs(dic[point][0]-dic[point][1]))

        thickness = np.array(thickness)
        return np.nanmean(thickness), np.nanstd(thickness), np.nanmax(thickness), np.nanmin(thickness)
    except:
        return np.nan, np.nan, np.nan, np.nan
        
    
def perimeter_calc(corcal):
    gray = np.uint8(corcal * 255)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # opencv 3.4.2
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # opencv 4.7.0.72
    cnt_len = cv2.arcLength(contours[0], True)
    return cnt_len


## Calculate the curve length of CC skeleton based on approximate ellipse
def curve_calc_ellipse(skeleton):
    try:
        # Get the min and max indices based on x-axis
        y1 = np.where(skeleton)[1][0]
        y2 = np.where(skeleton)[1][-1]
        x1 = np.where(skeleton)[0][0]
        x2 = np.where(skeleton)[0][-1]

        # Get the major axis length for ellipse
        xmid_cc, ymid_cc = (x1+x2)/2, (y1+y2)/2
        a = euclidean_dist(x1, xmid_cc, y1, ymid_cc)

        # Get all the points on the skeleton in different arrays
        x_arr = np.where(skeleton)[0]
        y_arr = np.where(skeleton)[1]

        # Get dot product to find the point on line normal to the major axis
        ls = {}
        for i in range(0, len(x_arr)):
            m1 = (ymid_cc-y1)/(xmid_cc-x1)
            m2 = (ymid_cc-y_arr[i])/(xmid_cc-x_arr[i])
            ls[(x_arr[i], y_arr[i])] = m1*m2

        # Get the index of the point closest to the normal line and find minor axis length
        # Get values less than equal to -1
        result = dict((k, v) for k, v in ls.items() if v <= -1)
        # Sort dictionary so that value closest to -1 is at the end
        sorted_result = sorted(result.items(), key=lambda kv: kv[1])
        # If the normal is parallel to the y axis
        if len(sorted_result) == 0:
            sorted_result = sorted(ls.items(), key=lambda kv: kv[1])
        # Get the x and y indices on the cc skeleton
        xnew = float(list(sorted_result)[-1][0][0])
        ynew = float(list(sorted_result)[-1][0][1])
        b = euclidean_dist(xmid_cc, xnew, ymid_cc, ynew)
        if a>b:
            curvature = a*a/b
        else:
            curvature = b*b/a

        # Arc length of semi-ellipse
        arc_length = calc_arc_length(a,b)

        # Angle at center of ellipse
        angle = calc_angle(x1,xnew,x2,y1,ynew,y2)

        return a, b, 1/curvature, arc_length, 180 + angle
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan


## Calculate the curve length of CC skeleton
def calc_geo_dist(skeleton):
    try:
        y_ = np.where(skeleton)[1]
        x_ = np.where(skeleton)[0]
        y1 = y_[0]
        y2 = y_[-1]
        x1 = x_[0]
        x2 = x_[-1]

        # Get only right side points
        x_right = []
        y_right = []
        for i in range(0, len(x_)):
            angle = calc_angle(x1,x2,x_[i],y1,y2,y_[i])
            if angle>=0.0:
                x_right.append(x_[i])
                y_right.append(y_[i]) 
                
        # Get the euclidean distance between consecutive points        
        tot = 0
        for i in range(0, len(x_right)-1):
            tot+=euclidean_dist(x_right[i], x_right[i+1], y_right[i], y_right[i+1])
        return tot
    except:
        return np.nan


## Calculate highest point on CC skeleton from the major axis vector of ellipse
def calc_max_ab(skeleton):
    try:
        y_ = np.where(skeleton)[1]
        x_ = np.where(skeleton)[0]
        y1 = y_[0]
        y2 = y_[-1]
        x1 = x_[0]
        x2 = x_[-1]
        m=(y2-y1)/(x2-x1)

        # Get only right side points
        x_right = []
        y_right = []
        for i in range(0, len(x_)):
            angle = calc_angle(x1,x2,x_[i],y1,y2,y_[i])
            if angle>=0.0:
                x_right.append(x_[i])
                y_right.append(y_[i])

        def shortest_distance(x1, y1, a, b, c):
            d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
            return d

        dist = []
        coords = []
        for i in range(len(x_right)):
            dist.append(shortest_distance(x_right[i], y_right[i], m, -1, y1-m*x1))
            y3 = (m*x_right[i] + m*m*y_right[i] + y1 - m*x1)/(1+m*m)
            x3 = (x_right[i] + m*y_right[i] - m*y1 + m*m*x1)/(1+m*m)
            coords.append([x3,y3])

        max_ind = np.argmax(dist)
        peak_a_dist = euclidean_dist(x1,coords[max_ind][0],y1,coords[max_ind][1])
        peak_b_dist = max(dist)

        return peak_a_dist, peak_b_dist
    except:
        return np.nan, np.nan


def pointwise_curve_calc(path_array):
    try:
        dx_dt = np.gradient(np.where(path_array)[0])
        dy_dt = np.gradient(np.where(path_array)[1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs((dx_dt*d2y_dt2 - dy_dt*d2x_dt2)/((dx_dt**2 + dy_dt**2)**(3/2)))

        return np.nanmean(curvature), np.nanstd(curvature), np.nanmax(curvature), np.nanmin(curvature)
    except:
        return np.nan, np.nan, np.nan, np.nan


def remove_skeleton_branches(skeleton):
    return

def count_connected_components(image):
    real_pt2 = np.zeros(shape = image.shape)
    for i,j in zip(np.where(image!=0)[0], np.where(image!=0)[1]):
        real_pt2[i][j] = 1

    _, count = skimage.measure.label(real_pt2, connectivity=None, return_num=True)
    return count


def get_shape_metrics(inp, subj, out):
    # create output directory
    os.makedirs(out, exist_ok=True)

    ## Preprocessing
    image  = extract_cc(inp + "/" + subj + ".nii.gz")
    thr001 = (image > 0.001).astype(int)
    labeled = label(thr001)
    
    area = 0
    cc_label = 0
    orientation = None
    for i, prop in enumerate(regionprops(labeled)):
        if prop.area > area:
            area = prop.area
            cc_label = i+1
            orientation = prop.orientation

    cc = (labeled == cc_label).astype(int)
    skeleton = thin(cc)
    rotate_by = np.sign(orientation)*(90 - np.abs(orientation) * 180 / np.pi)
    rotated = rotate(cc, angle=rotate_by, order=0, preserve_range=True)

    ## Subregion Segmentation
    y_min = np.min(np.where(rotated)[1])
    y_max = np.max(np.where(rotated)[1])
    straight_length = y_max - y_min

    # Witelson segmentation
    witelson5 = np.ones_like(rotated)
    witelson5[:,(y_max-int(1/2.*straight_length)):(y_max-int(1/3.*straight_length))] = 2
    witelson5[:,(y_max-int(2/3.*straight_length)):(y_max-int(1/2.*straight_length))] = 3
    witelson5[:,(y_max-int(4/5.*straight_length)):(y_max-int(2/3.*straight_length))] = 4
    witelson5[:,:y_max-int(4/5.*straight_length)] = 5

    rotated_witelson5 = rotate(witelson5, angle=-rotate_by, order=0, preserve_range=True)
    cc_witelson5 = cc * rotated_witelson5
    cc_witelson5_copy = cc * rotated_witelson5

    # JHU segmentation
    jhu3 = np.ones_like(rotated)
    jhu3[:,(y_max-int(5/6.*straight_length)):(y_max-int(1/6.*straight_length))] = 2
    jhu3[:,:y_max-int(5/6.*straight_length)] = 3

    rotated_jhu3 = rotate(jhu3, angle=-rotate_by, order=0, preserve_range=True)
    cc_jhu3 = cc * rotated_jhu3
    cc_jhu3_copy = cc * rotated_jhu3

    # Fix Genu Tips
    for atlas, num_regions in zip([cc_witelson5_copy, cc_jhu3_copy], [5, 3]):
        for i in range(2, num_regions+1):
            subregion = (atlas == i).astype(int)
            sublabels, num_labels = label(subregion, return_num=True)
            if num_labels > 1:
                to_convert = 0
                area_to_convert = 0
                for n in range(1, num_labels+1):
                    if (sublabels == n).astype(int).sum() > area_to_convert:
                        to_convert = n
                atlas[atlas == to_convert] = 1

    skel_witelson5 = skeleton * rotated_witelson5
    skel_jhu3 = skeleton * rotated_jhu3


    ## Metric Extraction
    mdict = {}

    # Whole region metrics
    mdict["Total_Area"] = np.sum(cc)   # Total area
    mdict["Total_a"], mdict["Total_b"], mdict["Total_Curvature"], mdict["Total_ArcLength"], mdict["Center_Angle"] = curve_calc_ellipse(skeleton) # Total curvature based on ellipse(mid point of 2a)
    mdict["Total_peak_a"], mdict["Total_peak_b"] = calc_max_ab(skeleton) # a and b based on highest point on CC skeleton
    mdict["Total_MeanThickness"], mdict["Total_StdThickness"], mdict["Total_MaxThickness"], mdict["Total_MinThickness"] = thickness_calc(rotated)    # Thickness  
    mdict["Total_GeodesicLength"] = calc_geo_dist(skeleton)  ## Geodesic path length for CC skeleton
    mdict["Total_Perimeter"] = perimeter_calc(rotated)   # Perimeter
    mdict["Total_MedialCurveLength"] = np.sum(skeleton)   # Medial curve length
    mdict["Ratio_MedialCurve_MaxEuclideanDist"] = mdict["Total_MedialCurveLength"]/mdict["Total_a"]  # Ratio of medial curve to max euclidean dist to measure hump
    mdict["Ratio_Geodesic_ArcLength"] = mdict["Total_GeodesicLength"]/mdict["Total_ArcLength"]  # Ratio of geodesic curve length to Total Arc Length to measure hump
    mdict["Hump_Distance"] = mdict["Total_a"] - mdict["Total_peak_a"]  ### if negative value: hump is after mid point
   
    # Whitelson5_Genu metrics
    mdict["Witelson5_Genu_Area"] = np.sum((cc_witelson5 == 1).astype(int))   #Area
    mdict["Witelson5_Genu_a"], mdict["Witelson5_Genu_b"], mdict["Witelson5_Genu_Curvature"], mdict["Witelson5_Genu_ArcLength"],_ = curve_calc_ellipse((skel_witelson5 == 1).astype(int))    # Curvature based on ellipse
    mdict["Witelson5_Genu_MeanThickness"], mdict["Witelson5_Genu_StdThickness"], mdict["Witelson5_Genu_MaxThickness"], mdict["Witelson5_Genu_MinThickness"] = thickness_calc((cc_witelson5 ==1).astype(int)) 

    # Witelson5_AnteriorBody metrics
    mdict["Witelson5_AnteriorBody_Area"] = np.sum((cc_witelson5 == 2).astype(int))
    mdict["Witelson5_AnteriorBody_a"], mdict["Witelson5_AnteriorBody_b"], mdict["Witelson5_AnteriorBody_Curvature"], mdict["Witelson5_AnteriorBody_ArcLength"],_ = curve_calc_ellipse((skel_witelson5 == 2).astype(int))
    mdict["Witelson5_AnteriorBody_MeanThickness"], mdict["Witelson5_AnteriorBody_StdThickness"], mdict["Witelson5_AnteriorBody_MaxThickness"], mdict["Witelson5_AnteriorBody_MinThickness"] = thickness_calc((cc_witelson5 ==2).astype(int)) 

    # Witelson5_PosteriorBody metrics
    mdict["Witelson5_PosteriorBody_Area"] = np.sum((cc_witelson5 == 3).astype(int))
    mdict["Witelson5_PosteriorBody_a"], mdict["Witelson5_PosteriorBody_b"], mdict["Witelson5_PosteriorBody_Curvature"], mdict["Witelson5_PosteriorBody_ArcLength"],_ = curve_calc_ellipse((skel_witelson5 == 3).astype(int))
    mdict["Witelson5_PosteriorBody_MeanThickness"], mdict["Witelson5_PosteriorBody_StdThickness"], mdict["Witelson5_PosteriorBody_MaxThickness"], mdict["Witelson5_PosteriorBody_MinThickness"] = thickness_calc((cc_witelson5 == 3).astype(int)) 

    # Witelson5_Isthmus metrics 
    mdict["Witelson5_Isthmus_Area"] = np.sum((cc_witelson5 == 4).astype(int))
    mdict["Witelson5_Isthmus_a"], mdict["Witelson5_Isthmus_b"], mdict["Witelson5_Isthmus_Curvature"], mdict["Witelson5_Isthmus_ArcLength"],_ = curve_calc_ellipse((skel_witelson5 == 4).astype(int))
    mdict["Witelson5_Isthmus_MeanThickness"], mdict["Witelson5_Isthmus_StdThickness"], mdict["Witelson5_Isthmus_MaxThickness"], mdict["Witelson5_Isthmus_MinThickness"] = thickness_calc((cc_witelson5 == 4).astype(int)) 

    # Witelson5_Splenium metrics
    mdict["Witelson5_Splenium_Area"] = np.sum((cc_witelson5 == 5).astype(int))
    mdict["Witelson5_Splenium_a"], mdict["Witelson5_Splenium_b"], mdict["Witelson5_Splenium_Curvature"], mdict["Witelson5_Splenium_ArcLength"],_ = curve_calc_ellipse((skel_witelson5 == 5).astype(int))
    mdict["Witelson5_Splenium_MeanThickness"], mdict["Witelson5_Splenium_StdThickness"], mdict["Witelson5_Splenium_MaxThickness"], mdict["Witelson5_Splenium_MinThickness"] = thickness_calc((cc_witelson5 == 5).astype(int)) 

    # JHU3_Genu metrics
    mdict["JHU3_Genu_Area"] = np.sum((cc_jhu3 == 1).astype(int))
    mdict["JHU3_Genu_a"], mdict["JHU3_Genu_b"], mdict["JHU3_Genu_Curvature"], mdict["JHU3_Genu_ArcLength"],_ = curve_calc_ellipse((skel_jhu3 == 1).astype(int))
    mdict["JHU3_Genu_MeanThickness"], mdict["JHU3_Genu_StdThickness"], mdict["JHU3_Genu_MaxThickness"], mdict["JHU3_Genu_MinThickness"] = thickness_calc((cc_jhu3 == 1).astype(int)) 

    # JHU3_Body metrics
    mdict["JHU3_Body_Area"] = np.sum((cc_jhu3 == 2).astype(int))
    mdict["JHU3_Body_Genu_a"], mdict["JHU3_Body_Genu_b"], mdict["JHU3_Body_Curvature"], mdict["JHU3_Body_ArcLength"],_ = curve_calc_ellipse((skel_jhu3 == 2).astype(int))
    mdict["JHU3_Body_MeanThickness"], mdict["JHU3_Body_StdThickness"], mdict["JHU3_Body_MaxThickness"], mdict["JHU3_Body_MinThickness"] = thickness_calc((cc_jhu3 == 2).astype(int)) 

    # JHU3_Splenium metrics
    mdict["JHU3_Splenium_Area"] = np.sum((cc_jhu3 == 3).astype(int))
    mdict["JHU3_Splenium_a"], mdict["JHU3_Splenium_b"], mdict["JHU3_Splenium_Curvature"], mdict["JHU3_Splenium_ArcLength"],_ = curve_calc_ellipse((skel_jhu3 == 3).astype(int))
    mdict["JHU3_Splenium_MeanThickness"], mdict["JHU3_Splenium_StdThickness"], mdict["JHU3_Splenium_MaxThickness"], mdict["JHU3_Splenium_MinThickness"] = thickness_calc((cc_jhu3 == 3).astype(int)) 
    mdict["number_of_pieces"] = count_connected_components(image)
    
    # Write Output Spreadsheet
    metrics = pd.DataFrame.from_dict(mdict, orient='index')
    metrics.to_csv(out + "/" + subj + ".csv", index_label="Measures", header=['Value'])
