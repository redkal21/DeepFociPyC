import os, sys, email, re
from nltk.corpus import stopwords
import glob
from glob import glob
from os.path import join
import time
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import transforms
import keras
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import array_to_img
from keras.preprocessing import image
import shutil
import read_roi
from skimage import io, feature, filters, morphology, draw
from skimage.measure import regionprops
import skimage.measure as measure
from skimage.measure import shannon_entropy
from skimage.measure import blur_effect
from skimage.measure import euler_number
from skimage.measure import subdivide_polygon
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.datasets import load_digits
from umap import UMAP
import plotly.express as px
from sklearn.utils import class_weight
from sklearn.ensemble import VotingClassifier
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
import time
from sklearn import model_selection
from sklearn.model_selection import cross_validate
import statistics
from sklearn.utils import class_weight

# check version number
import imblearn
from sklearn import datasets
import sklearn.neighbors
from scipy import stats
import itertools
from sklearn.metrics import roc_curve
from itertools import cycle
from cycler import cycler
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import math
from math import log, pi, sqrt
from yellowbrick.classifier import ClassPredictionError
import numpy as np
import pandas as pd
import cv2
import os
import mahotas as mh
# from __future__ import division #to avoid integer division problem

import seaborn as sns
from itertools import repeat
# for machine learning focus detection
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


def focus_classifier(unclassified_foci_props):
    foci_info_file = 'Z:/rothenberglab/archive/Maria/2020_FociData' + '/foci_info.csv'
    df_foci_full = pd.read_csv(foci_info_file)
    foci_classes = df_foci_full['status'].unique()
    df_foci = df_foci_full.copy()
    # change other focus classes that aren't 1 (true focus) to 0 (false focus)
    df_foci.loc[(df_foci.status > 1), ('status')] = 0

    # split data into train and test set
    X_raw = np.array(
        list([df_foci['area'].to_numpy(), df_foci['perimeter'].to_numpy(), df_foci['major_axis_length'].to_numpy(),
              df_foci['minor_axis_length'].to_numpy()]))
    Y_raw = np.array(list(df_foci['status']))
    X_notscaled, test_X, Y_notscaled, test_y = train_test_split(X_raw.transpose(), Y_raw, test_size=0.3, random_state=0)

    # scale data to speed up svm
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_notscaled)  # MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X_notscaled)  # scaling.transform(X_raw.transpose())
    test_X = scaling.transform(test_X)  # scaling.transform(test_X)
    Y = Y_notscaled

    # scale input data
    unclassified_foci_props = unclassified_foci_props.transpose()
    scaling_ = MinMaxScaler(feature_range=(-1, 1)).fit(unclassified_foci_props)
    unclassified_foci_props_sc = scaling_.transform(unclassified_foci_props)

    # initialize SVM
    clf = svm.SVC(kernel='poly', gamma=10, C=0.05)
    # fit the model with the training data
    clf.fit(X, Y)
    # check accuracy of model
    accuracy = clf.score(test_X, test_y)

    # predict class of focus for detected foci
    classified_foci = clf.predict(unclassified_foci_props_sc)

    return classified_foci


def draw_on_img(file, file_root, draw_img, obj_bbox, obj_coords, center_coordinates, box, l, w, draw_color,
                draw_bbox=True, draw_contours=True, draw_label=True, label=''):
    # draw tracking window (bbox) and object contours on intensity stack in white color
    # if draw_label == True then also label will be drawn at top corner of bbox
    # imgplot = plt.imshow(draw_img) # img_rgb_1
    # plt.show()
    # plt.close()

    (min_row, min_col, max_row, max_col) = obj_bbox
    if (draw_bbox):
        cv2.rectangle(draw_img, (min_col, min_row), (max_col, max_row), draw_color, 1)
    if (draw_label == True):
        test_img = draw_img
    cv2.putText(test_img, label, (max_col, max_row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 1,
                cv2.LINE_AA)  # before draw_color: 0.5
    # cv2.putText(draw_img, label, (max_col, max_row), cv2.FONT_HERSHEY_SIMPLEX, 0.23, draw_color, 1, cv2.LINE_AA)
    # im = Image.new(mode='RGB', size=(draw_img.shape[0], draw_img.shape[1]))
    # draw = ImageDraw.Draw(draw_img)
    # font = ImageFont.truetype("arial.ttf", 18)
    # draw.text((max_col, max_row), label, (255, 255, 255), font=font)

    # imgplot = plt.imshow(draw_img) # img_rgb_1
    # plt.show()
    # plt.close()

    if (draw_contours):
        coords = obj_coords
    obj_mask = np.zeros(shape=draw_img.shape, dtype='uint8')  # shape=draw_img.shape[:-1]
    #Loops through all coordinates for each image applying object masks - image processing function
    #Draws contours around detected regions of interests with r = 1, blue color
    for i in range(len(coords)):
        obj_mask[coords[i][0], coords[i][1]] = 255
    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(draw_img, contours, -1, draw_color, 1)
    # Radius of circle
    radius = 1
    # Blue color in BGR
    color = draw_color
    # color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    # cv2.circle(draw_img, center_coordinates, radius, color, thickness)


# folder format is DATE_DRUG_TIME
def get_dir_list(base_dir):
    final_dir_dict = {}
    # get all directories that match expected format
    dir_list = glob(base_dir + '/*')
    #Loops through public directories and scans for folders formatted in DATE_DRUG_TIME
    for dir_ in dir_list:
        if (os.path.isdir(dir_)):
            key = os.path.split(dir_)[1]
        mo = re.match(r'^([0-9]+)_([^_]+)_([^_]+)$', key)
        if (not mo):
            continue
    # skipping, directory name is not correct format

    date = mo.group(1)
    drug = mo.group(2)
    treatment_time = mo.group(3)

    final_dir_dict[key] = []

    cur_dir = dir_ + '/Output'
    experiment_dirs = glob(cur_dir + '/*')
    """Loops through experimental directories and looks for a match that consists of only 
    digits (r’ ^[0-9]+$’ - reading for names that contain adjacent / contiguous string of digits), 
    if it doesn’t find a match for digit-specific directories, it skips"""
    for exp_dir in experiment_dirs:
        if (os.path.isdir(exp_dir)):
            mo = re.match(r'^[0-9]+$', os.path.split(exp_dir)[1])
        if (not mo): continue

    exp_num = os.path.split(exp_dir)[1]
    spool_dirs = glob(exp_dir + '/spool_*')
    #spool refers to temporary storage spaces
    #Under exp directories, there are spool + other directories
    """Prior to for loop, created another global spool ‘dirs’ to include exp + spool directories;
    for loop scans for directory names with spool + digits (‘[0-9]+$’); no match = skip final_dir_dict;
    Collects list of matched directories with formatted dates, drugs, etc. specs;
    Prints list of spool directories captured in final directory dictionary"""
    for spool_dir in spool_dirs:
        if (os.path.isdir(spool_dir)):
            spool = os.path.split(spool_dir)[1]
        mo = re.match(r'^spool_[0-9]+$', spool)
        if (not mo):
            continue
    # print(spool_dir)
        final_dir_dict[key].append([date, drug, treatment_time, exp_num, spool, spool_dir])
        return final_dir_dict


def pre_process_movies(summary_dir, dir_list):
    # dir_list = dir_list[5:8] # for
    #Pre - process function; two parameters(summary_dir, dir_list)
    """Reads BP1 - 2 data and to the dir_list with 25 elements (  # 25 is randomly selected for debugging);
    # its loops through and adds sub-folder extension ‘/lefts’ + ‘/rights’  to 5 of the dir_ elements each
    once labeled, the grayed out indicated location of left and right channel directories in
    local machine (left_chnl_dir_files is a variable, and os.listdir(left_chnl_dir) is the location
    where this directory is locally stored)"""
    """dir holds the name to each directory in the list, not the whole element when calling dir_5, only accessing
    5th element in the file's name"""
    for dir_ in dir_list:
        # Read BP1-2 data
        # dir_ = dir_list[25] # for debugging
        left_chnl_dir = dir_[5] + '/lefts'
        right_chnl_dir = dir_[5] + '/rights'
        left_chnl_dir_files = os.listdir(left_chnl_dir)
        right_chnl_dir_files = os.listdir(right_chnl_dir)

    # import images
    def last_4chars(x):
        print(x[-8:])
        return (x[-8:])

    # Compile images; left = red chnl, right, green chnl
    # Define directories
    spool_list = sorted(left_chnl_dir_files)
    single_left_file = io.imread(left_chnl_dir + '/' + spool_list[1])
    r_raw, c_raw = single_left_file.shape
    spool_list = spool_list[0:3]
    # file = spool_list # if only using first frame
    # Initialize image
    rgb_movie = np.empty(shape=((len(spool_list)), r_raw, c_raw, 3), dtype='uint16') * 0  #
    t_ = 0

    # Import images from each channel
    # Use if more than 1st image is used
    for file in spool_list:
        """within the context of the created spool_list directory its looping through and asking python,
        if the file is a thumbs.db, which is a database file containing a small JPEG image representing the larger file
        then it will continue to the next file in the spool_list"""
        if (file == 'Thumbs.db'):
            continue
        """if not, it will read the file and then saves it to both lists after creating a new subdirectory under right
        and left channels and increment t_ with parameters - a way to track # of files that are not 'thumbs.db' """
        # after increments, the new file is saved within a new subdirectory
        rgb_movie[t_, :, :, 0] = io.imread(left_chnl_dir + '/' + file)
        rgb_movie[t_, :, :, 1] = io.imread(right_chnl_dir + '/' + file)
        t_ = t_ + 1
        io.imsave(dir_[5] + '/' + 'rgb_movie.tif', rgb_movie)

        # Find ROI file
        # single ROI
        roi_list = glob(dir_[5] + '/*.roi')
        #count # of roi files in roi_list and if length is 1, then there isn't a zipped file
        #if the length is less than 1, then there are no files in this list, so it scans for a zipped file
        #for lengths beyond these conditions, it will print message and it will continue to the next iteration
        if (len(roi_list) == 1):
            roi_is_zip = False
        elif (len(roi_list) < 1):
            # more than one ROI
            roi_list = glob(dir_[5] + '/*.zip')
            roi_is_zip = True
        else:
            print("Missing ROI file for movie file")
            continue

    for roi_file in roi_list:
        print('Processing ', roi_file)
        # load the ROI and extract the coords
        # make a mask from the coords
        # create folder for this cropped out cell
        # use mask to save the cropped movie file
        """if its zipped file, then the zipped file will be read and stored as roi_points"""
        if (roi_is_zip):
            roi_points = read_roi.read_roi_zip(roi_file)
            """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
            fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
        else:
            fobj = open(roi_file, 'r+b')
            roi_points = read_roi.read_roi(fobj)
            roi_points = [roi_points, ]  # make format same as if its a zip

            # ROI is bounding box, make mask
            # go through each frame and apply masks, save cropped movie file
            num_frames = len(rgb_movie)
            "based on file coords, roi_points = roi_count, bbox_points = second-coord, add # values to refer to file types"
    for roi_count, bbox_points in enumerate(roi_points):
            # if this "os.path..." file DNE, then it will mkdir, meaning make a new directory with this pathway

        """regardless of creation or not of the directory, then it will continue through loop, calc steps and columns
        + more image-processing steps; shape within np.empty function determines multi-dimensionality - 4 elements in ()s
        = 4-dimensional"""
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1))):
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1))
            n_rows = bbox_points[2][0] - bbox_points[0][0] + 1
            n_cols = bbox_points[2][1] - bbox_points[0][1] + 1
            cropped_rgb_movie = np.empty(shape=(num_frames, n_rows, n_cols, 3), dtype='uint16')
            cropped_g_movie = np.empty(shape=(num_frames, n_rows, n_cols, 1), dtype='uint16')
            raw_cropped_g_movie = np.empty(shape=(num_frames, n_rows, n_cols, 1), dtype='uint16')
            file_root = os.path.split(file)[1][:-4]
    """looping through frames in range of num_frames and add roi point coordinates to each frame and then given back to
    the frame_i to reshape/recolor2 it; rgb - red, green, blue vs grayscale - rgb vs g"""
    for frame_i in range(num_frames):
        # append = adding to the end
        # 0.0 append frames and extract defined rois
        cropped_rgb_movie[frame_i] = rgb_movie[frame_i][bbox_points[0][0]:bbox_points[2][0] + 1,
                                         bbox_points[0][1]:bbox_points[2][1] + 1, :]
        cropped_g_movie[frame_i, :, :, :] = cropped_rgb_movie[frame_i, :, :, 1].reshape((n_rows, n_cols, 1))

      # 0.1 extract defined roi
    raw_img = cropped_g_movie[frame_i, :, :, :].reshape(n_rows, n_cols).copy()
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
     4] + '_preproc_01_raw_single_frame_roi' + str(roi_count + 1) + '_.tif', raw_img)

    # 0.2 histogram equalization
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(10, 10))
    img = clahe.apply(raw_img)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_02_clahe_roi' + str(roi_count + 1) + '_.tif', img)

    # 0.3 remove hot pixels
    medblurred_img = cv2.medianBlur(img, 3)
    dif_img = img - medblurred_img
    img = img - dif_img
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_03_hotpixfilt_roi' + str(roi_count + 1) + '_.tif', img)

    # 0.4 gaussian blur
    img = cv2.GaussianBlur(img, (7, 7), 0)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_04_gaussblur_roi' + str(roi_count + 1) + '_.tif', img)

    # 0.5 background subtraction via k means clustering
    max_pix = np.max(img)
    Z = img.reshape((n_rows * n_cols), 1)
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4  # 3 # for #1: 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    sorted_center = center[:, 0].sort()
    _, img = cv2.threshold(img, int(center[2]), max_pix, cv2.THRESH_TOZERO)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_05_bskmeans_roi' + str(roi_count + 1) + '_.tif', img)

    # append processed single roi for this frame
    cropped_g_movie[frame_i, :, :, :] = np.asarray(img).reshape(n_rows, n_cols, 1)
    raw_cropped_g_movie[frame_i, :, :, :] = np.asarray(raw_img).reshape(n_rows, n_cols, 1)

    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_roi' + str(roi_count + 1) + '_.tif', cropped_g_movie)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_raw_roi' + str(roi_count + 1) + '_.tif', raw_cropped_g_movie)

    # 1.0 mean projection
    max_img = np.mean(cropped_g_movie, axis=0).reshape(n_rows, n_cols)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_10_maxproj_roi' + str(roi_count + 1) + '_.tif', max_img.astype('uint16'))
    raw_max_img = np.mean(raw_cropped_g_movie, axis=0).reshape(n_rows, n_cols)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_10_rawmaxproj_roi' + str(roi_count + 1) + '_.tif', raw_max_img.astype('uint16'))

    # 1.1 OTSU thresholding
    th = filters.threshold_otsu(max_img)
    img_mask = max_img > th
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_11_OTSU_roi' + str(roi_count + 1) + '_.tif', img_mask)

    # 1.2 connected component labeling
    l_, n_ = mh.label(img_mask.reshape(n_rows, n_cols), np.ones((3, 3), bool))  # binary_closed_hztl_k
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_12_ccl_roi' + str(roi_count + 1) + '_.tif', l_)

    # 1.3 measure region properties
    rs_k = regionprops(l_)
    im_props = regionprops(l_, intensity_image=max_img.reshape(n_rows, n_cols))
    results = []

        #scanning for desired blob sizes less than 25 and greater 1000, then if size is within range
        #then it will skip to the next blob
    for blob in im_props:
        properties = []
        if ((blob.area < 25) or (blob.area > 1000)):
            continue

    # blob = im_props[30]
    properties.append(blob.label)
    properties.append(blob.centroid[0])
    properties.append(blob.centroid[1])
    properties.append(blob.orientation)
    properties.append(blob.area)
    properties.append(blob.perimeter)
    properties.append(blob.major_axis_length)
    properties.append(blob.minor_axis_length)
    properties.append(blob.eccentricity)
    properties.append(blob.coords)
    properties.append(blob.bbox)

    i = blob.label
    test = blob.coords
    x, y, w, h = cv2.boundingRect(test)
    single_focus_img = max_img[x:x + w, y:y + h]
    single_focus_img_raw = raw_max_img[x:x + w, y:y + h]
    rows, cols = single_focus_img_raw.shape
    (h, w) = single_focus_img_raw.shape

    # ------------------------------------
    # blur detection in IQM
    abs_f = np.abs(f)
    mag_max = np.max(abs_f)
    thresh = mag_max / 1000
    thresh_f = (abs_f < thresh).sum()
    imagequalmeas = thresh_f / (rows * cols)
    # ------------------------------------
    # save images
    # grayscale
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_focusNo' + str(i) + '_0_grayscale' + str(roi_count + 1) + '.tif',
              single_focus_img_raw.astype('uint8'))

    # gaussian blur
    gaussblur_img = cv2.GaussianBlur(single_focus_img_raw.astype('uint8'), (7, 7), 0)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_focusNo' + str(i) + '_gausblur_roi' + str(roi_count + 1) + '.tif', gaussblur_img)
    otsu_th = filters.threshold_otsu(gaussblur_img)
    _, thresh_gauss_img = cv2.threshold(gaussblur_img, otsu_th, 255, cv2.THRESH_TOZERO)
    single_focus_img_gray = single_focus_img / 4
    single_focus_img_gray_8U = single_focus_img_gray.astype('uint8')
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[4] + str(
        roi_count + 1) + 'full_max.tif', max_img.astype('uint8'))
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_focusNo' + str(i) + '_roi' + str(roi_count + 1) + '.tif', single_focus_img_gray_8U)

    proj_img_x = np.mean(single_focus_img, axis=0).reshape(w, )
    proj_img_y = np.mean(single_focus_img, axis=1).reshape(h, )

    l_x = len(proj_img_x)
    l_y = len(proj_img_y)
    # if the length along the x-axis is not equal to the length along y, then check if y is greater than
    # x, if so, calculate the difference => length y - length x and store as proj_img_x after using numpy
    if (l_x != l_y):
        if (l_y > l_x):
            diff = l_y - l_x
            proj_img_x = np.pad(proj_img_x, [(0, diff)])

    if (l_x > l_y):
        diff = l_x - l_y
        proj_img_y = np.pad(proj_img_y, [(0, diff)])

    #proj_img_x = proj_img_x
    #proj_img_y = proj_img_y
    labels_x = np.arange(0, len(proj_img_x), 1)

    fig = plt.figure()
    width = 1
    ax_x = plt.subplot(3, 1, 1)
    ax_x.bar(labels_x, proj_img_x, width, color='g', edgecolor="k")
    ax_x.set_xlabel("")
    ax_x.set_title(' x ')

    labels_y = np.arange(0, len(proj_img_y), 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_y.bar(labels_y, proj_img_y, width, color='g', edgecolor="k")
    ax_y.set_xlabel("")
    ax_y.set_title(' y ')

    ax_xy = plt.subplot(3, 1, 3)
    rects1 = ax_xy.bar(labels_x, proj_img_x, width, color='royalblue', edgecolor="k")
    rects2 = ax_xy.bar(labels_y, proj_img_y, width, color='seagreen', edgecolor="k")
    ax_xy.set_xlabel("")
    ax_xy.legend((rects1[0], rects2[0]), ('x', 'y'))
    plt.close(fig)

    proj_img_x_ = proj_img_x.ravel().astype('float32')
    proj_img_y_ = proj_img_y.ravel().astype('float32')

    compare_val_correl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CORREL)
    compare_val_chisq_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR)
    compare_val_intrsct_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_INTERSECT)
    compare_val_bc_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_BHATTACHARYYA)
    compare_val_chisqalt_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR_ALT)
    compare_val_hellinger_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_HELLINGER)
    compare_val_kl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_KL_DIV)

    properties.append(imagequalmeas)
    properties.append(compare_val_correl_xy)
    properties.append(compare_val_chisq_xy)
    properties.append(compare_val_intrsct_xy)
    properties.append(compare_val_bc_xy)
    properties.append(compare_val_chisqalt_xy)
    properties.append(compare_val_hellinger_xy)
    properties.append(compare_val_kl_xy)
    results.append(properties)

    df_foci_full = pd.DataFrame(results,
                                columns=['focus_label', 'centroid-0', 'centroid-1', 'orientation', 'area',
                                         'perimeter', 'major_axis_length',
                                         'minor_axis_length', 'eccentricity', 'coords', 'bbox', 'img_qual_meas',
                                         'var_laplc_raw_otsu', 'var_laplc_raw', 'var_laplc',
                                         'compare_val_correl',
                                         'compare_val_chisq', 'compare_val_intrsct', 'compare_val_bc',
                                         'compare_val_chisqalt', 'compare_val_hellngr', 'compare_val_kl'])

    # save labeled and contoured foci
    a = img.copy()
    max_val = int(np.max(a))
    raw_max_val = int(np.max(raw_max_img))
    center_coordinates = 0
    box = 0
    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        draw_on_img(file, file_root, a, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1,
                    (max_val, max_val, max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True,
                    label=str(focus_full_row['focus_label']))  # str(focus_full_i)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_procfocusDetectionCheck_Full.tif', a)

    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        draw_on_img(file, file_root, raw_max_img, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1,
                    (raw_max_val, raw_max_val, raw_max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True,
                    label=str(focus_full_row['focus_label']))  # str(focus_full_i)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_rawfocusDetectionCheck_Full.tif', raw_max_img.astype('uint16'))
    stop = 1
    print("Done.")
    stop = 1


def pre_process_movies_idr(summary_dir, dir_list):
    # dir_list = dir_list[14:16] # for debugging
    # dir_list = [dir_list[i] for i in [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23]]
    """Pre - process - idr
    Reads from IDR data, and dir_files is the box / folder hosting location of 5th directory in local storage
    Sorting contents within new dir_files and sorted is now label spool_list
    Isolate the first element in spool_list and label spool_list_file
    Test is creating pathways to read each data set"""
    # potential change  - swap out (dir_[5] + '/' + spool_list_file  for test defined in prior line, 522)
    for dir_ in dir_list:
        # Read IDR data
        dir_files = os.listdir(dir_[5])
        spool_list = sorted(dir_files)
        spool_list_file = spool_list[0]
        # test = dir_[5] + '/' + spool_list_file #
        idr_file = io.imread(dir_[5] + '/' + spool_list_file)

    # import images
    def last_4chars(x):
        print(x[-8:])
        return (x[-8:])

    # Compile images; left = red chnl, right, green chnl
    # Define directories
    r_raw, c_raw = idr_file.shape
    file = spool_list_file  # if only using first frame
    # Initialize image
    # rgb_movie = np.empty(shape=((len(spool_list)), r_raw, c_raw, 3), dtype='uint16') * 0 #
    t_ = 0

    # Find ROI file
    # single ROI
    roi_list = glob(dir_[5] + '/' + spool_list_file[:-4] + '_roi' + '*.zip')

    # check number of rois
    roi_is_zip = True
    rgb_movie = idr_file

    """looping from roi_list and printing on screen 'Processing' for each roi_file"""
    for roi_file in roi_list:
        print('Processing ', roi_file)
    # load the ROI and extract the coords
    # make a mask from the coords
    # create folder for this cropped out cell
    # use mask to save the cropped movie file
    """if its zipped file, then the zipped file will be read and stored as roi_points"""
    if (roi_is_zip):
        roi_points = read_roi.read_roi_zip(roi_file)
        """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
            fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
    else:
        fobj = open(roi_file, 'r+b')
        roi_points = read_roi.read_roi(fobj)
        roi_points = [roi_points, ]  # make format same as if its a zip

    # ROI is bounding box, make mask
    # go through each frame and apply masks, save cropped movie file
    num_frames = len(rgb_movie)
    # roi_points = roi_points[35:37]# for debugging
    """roi_count iterates for roi_points, and bbox_points iterates for second element
    bb = bounding box, and enumerate associates incremental value to each roi_point"""
    for roi_count, bbox_points in enumerate(roi_points):
        # roi_count = 7 # for debugging
        print('roi count: ' + str(roi_count))
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1))):
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1))

        """if such-named directory is there, then itll be removed w shutil.rmtree - deleting the whole directory including subs;
        if there is no such directory, then itll be created with the mkdir function (for both '/DetectedFoci' and 'DetectedFoci_Outlines)"""
        if (os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')):  # delete individual foci directory
            shutil.rmtree(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')):  # create individual foci directory
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')
        if (os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines')):  # delete individual foci directory with outlines
            shutil.rmtree(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines')
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines')):  # create individual foci directory with outlines
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines')

    n_rows = bbox_points[2][0] - bbox_points[0][0] + 1
    n_cols = bbox_points[2][1] - bbox_points[0][1] + 1
    file_root = os.path.split(file)[1][:-4]

    for frame_i in range(num_frames):
        # 0.0 append frames and extract defined rois
        cropped_img = rgb_movie[bbox_points[0][0]:bbox_points[2][0] + 1, bbox_points[0][1]:bbox_points[2][1] + 1]

    # rescale image
    '''
    rescale_factor = 4
    cropped_img = cv2.resize(cropped_img, (0, 0), fx=rescale_factor, fy=rescale_factor)
    n_rows_rescale, n_cols_rescale = cropped_img.shape
    '''

    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_01_raw_roi' + str(roi_count + 1) + '_beforeResize.tif', cropped_img)

    # resize
    ratio = 5.02  # idr: 10.2 , df: 1.47
    width = int(cropped_img.shape[1] * ratio)
    height = int(cropped_img.shape[0] * ratio)
    dim = (width, height)
    cropped_img = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_LINEAR)
    n_rows_rescale, n_cols_rescale = cropped_img.shape
    len_img_test_new = cropped_img.size

    # 0.1 extract defined roi
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_01_raw_roi' + str(roi_count + 1) + '_afterResize.tif', cropped_img)

    img = cropped_img

    # 0.2 histogram equalization
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1, 1))
    img = clahe.apply(cropped_img)
    img_clahe = img.copy()
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_02_clahe_roi' + str(roi_count + 1) + '.tif', img)

    # 0.3 remove hot pixels
    medblurred_img = cv2.medianBlur(img, 3)
    dif_img = img - medblurred_img
    img = img - dif_img
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_03_hotpixfilt_roi' + str(roi_count + 1) + '.tif', img)

    # 0.4 gaussian blur
    img = cv2.GaussianBlur(img, (1, 1), 0)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_04_gaussblur_roi' + str(roi_count + 1) + '.tif', img)

    # 0.5 background subtraction via k means clustering
    max_pix = np.max(img)
    Z = img.reshape((n_rows_rescale * n_cols_rescale), 1)
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4  # 3 # for #1: 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    sorted_center = center[:, 0].sort()
    _, img = cv2.threshold(img, int(center[3]), max_pix, cv2.THRESH_TOZERO)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_05_bskmeans_roi' + str(roi_count + 1) + '.tif', img)

    # 1.0 OTSU thresholding
    th = filters.threshold_otsu(img)
    img_mask = img > th
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_11_OTSU_roi' + str(roi_count + 1) + '.tif', img_mask)

    # 1.1 connected component labeling
    l_, n_ = mh.label(img.reshape(n_rows_rescale, n_cols_rescale), np.ones((3, 3), bool))  # binary_closed_hztl_k
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_proc_12_ccl_roi' + str(roi_count + 1) + '.tif', l_)

    # 1.2 measure region properties
    rs_k = regionprops(l_)
    im_props = regionprops(l_, intensity_image=img.reshape(n_rows_rescale, n_cols_rescale))
    results = []
    """scanning blobs in im_props folder and labels, stores coordinates, draws rectangle on rois with cv2.boundingRect,
    applies focus tool, crops image"""
    #x:x + w - x is the first coord, x+w is the sum of the width across axis, and this operation calcs width of roi
    #y:y + h - calcs height, reads as y and y+h
    #focus_tot_pix - calcs # of pixels by order of area of roi
    for blob in im_props:
        properties = []
        i_focus = blob.label
        focus_coords = blob.coords
        x, y, w, h = cv2.boundingRect(focus_coords)
        single_focus_img = img[x:x + w, y:y + h]
        single_focus_img_raw = cropped_img[x:x + w, y:y + h]
        rows, cols = single_focus_img_raw.shape
        focus_tot_pix = rows * cols
        #If area of pixels (r*c) is less than 3 or greater than 1700, then it will skip to next blob in list of im_props
        #scanning for images between range; 3>pix<1700
        if ((focus_tot_pix < 3) or (focus_tot_pix > 1700)):  #
            continue

        # blob = im_props[30] # for debugging
        print('blob label: ' + str(blob.label))
        properties.append(blob.label)
        properties.append(blob.centroid[0])
        properties.append(blob.centroid[1])
        properties.append(blob.orientation)
        properties.append(blob.area)
        properties.append(blob.perimeter)
        properties.append(blob.major_axis_length)
        properties.append(blob.minor_axis_length)
        properties.append(blob.eccentricity)
        properties.append(blob.coords)
        properties.append(blob.bbox)

        focus_coords = blob.coords

        i = blob.label
        test = blob.coords
        x, y, w, h = cv2.boundingRect(test)
        single_focus_img = img[x:x + w, y:y + h]
        single_focus_img_raw = cropped_img[x:x + w, y:y + h]
        rows, cols = single_focus_img_raw.shape
        (h, w) = single_focus_img_raw.shape

        proj_img_x = np.mean(single_focus_img, axis=0).reshape(w, )
        proj_img_y = np.mean(single_focus_img, axis=1).reshape(h, )

        l_x = len(proj_img_x)
        l_y = len(proj_img_y)
        if (l_x != l_y):
            if (l_y > l_x):
                diff = l_y - l_x
        proj_img_x = np.pad(proj_img_x, [(0, diff)])
        #potential redundancy - if the length is not equal, find the difference; then the next chunk instructs that this code run regardless of relationship
        if (l_x > l_y):
            diff = l_x - l_y
        proj_img_y = np.pad(proj_img_y, [(0, diff)])

        proj_img_x = proj_img_x
        proj_img_y = proj_img_y
        labels_x = np.arange(0, len(proj_img_x), 1)

        fig = plt.figure()
        width = 1
        ax_x = plt.subplot(3, 1, 1)
        ax_x.bar(labels_x, proj_img_x, width, color='g', edgecolor="k")
        ax_x.set_xlabel("")
        ax_x.set_title(' x ')

        labels_y = np.arange(0, len(proj_img_y), 1)
        ax_y = plt.subplot(3, 1, 2)
        ax_y.bar(labels_y, proj_img_y, width, color='g', edgecolor="k")
        ax_y.set_xlabel("")
        ax_y.set_title(' y ')

        ax_xy = plt.subplot(3, 1, 3)
        rects1 = ax_xy.bar(labels_x, proj_img_x, width, color='royalblue', edgecolor="k")
        rects2 = ax_xy.bar(labels_y, proj_img_y, width, color='seagreen', edgecolor="k")
        ax_xy.set_xlabel("")
        ax_xy.legend((rects1[0], rects2[0]), ('x', 'y'))
        plt.close(fig)

        proj_img_x_ = proj_img_x.ravel().astype('float32')
        proj_img_y_ = proj_img_y.ravel().astype('float32')

        compare_val_correl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CORREL)
        compare_val_chisq_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR)
        compare_val_intrsct_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_INTERSECT)
        compare_val_bc_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_BHATTACHARYYA)
        compare_val_chisqalt_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR_ALT)
        compare_val_hellinger_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_HELLINGER)
        compare_val_kl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_KL_DIV)

        # pixel number
        len_single_focus = single_focus_img.size

        properties.append(compare_val_correl_xy)
        properties.append(compare_val_chisq_xy)
        properties.append(compare_val_intrsct_xy)
        properties.append(compare_val_bc_xy)
        properties.append(compare_val_chisqalt_xy)
        properties.append(compare_val_hellinger_xy)
        properties.append(compare_val_kl_xy)
        properties.append(len_single_focus)
        results.append(properties)

        # save images of individual foci, with and without outlines
        # extract focus bounding box
        focus_bbox = blob.bbox
        img_r1, img_c1 = img_clahe.shape
        # define bounding box
        pixel_border = 3
        r0 = focus_bbox[0] - pixel_border
        r1 = focus_bbox[2] + pixel_border
        c0 = focus_bbox[1] - pixel_border
        c1 = focus_bbox[3] + pixel_border
        # check if bounding box goes beyond image limits
        if (r0 < 0):
            r0 = 0
        if (r1 > img_r1):
            r1 = img_r1
        if (c0 < 0):
            c0 = 0
        if (c1 > img_c1):
            c1 = img_c1
        # focus with no outlines
        img_focus_clahe = img_clahe.copy()[r0:r1, c0:c1]
        # focus with outlines
        # draw outlines on full nucleus ROI
        focus_max_val = int(np.max(img_clahe) / 2)
        obj_mask = np.zeros(shape=img_clahe.shape, dtype='uint8')  # shape=draw_img.shape[:-1]
    """xyz"""
    for i in range(len(focus_coords)):
        obj_mask[focus_coords[i][0], focus_coords[i][1]] = 200
    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_clahe, contours, -1, (focus_max_val, focus_max_val, focus_max_val), 1)
    # crop out focus of interest
    img_focus_clahe_outlines = img_clahe[r0:r1, c0:c1]

    # save focus in focus folder
    # max proj
    # no outlines
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci/' + dir_[0] + '_' +
              dir_[1] + '_' + dir_[2] + '_' + dir_[3] + dir_[4] + '_roi' + str(roi_count + 1) +
              '_focusNo' + str(i_focus) + '_' + '.tif', img_focus_clahe)
    # outlines
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines/' + dir_[0] + '_' +
              dir_[1] + '_' + dir_[2] + '_' + dir_[3] + dir_[4] + '_roi' + str(roi_count + 1) +
              '_focusNo' + str(i_focus) + '_' + '.tif',
              img_focus_clahe_outlines)
# processing and cleaning up input of 2000 images and saving as new file
    df_foci_full = pd.DataFrame(results,
                                columns=['focus_label', 'centroid-0', 'centroid-1', 'orientation', 'area',
                                         'perimeter', 'major_axis_length',
                                         'minor_axis_length', 'eccentricity', 'coords', 'bbox',
                                         'compare_val_correl',
                                         'compare_val_chisq', 'compare_val_intrsct', 'compare_val_bc',
                                         'compare_val_chisqalt', 'compare_val_hellngr', 'compare_val_kl', 'tot_pix_no'])

    # save properties
    df_foci_full.to_csv(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + 'focus_info_full.csv')

    # save labeled and contoured foci
    a = img.copy()
    max_val = int(np.max(a))
    raw_max_val = int(np.max(cropped_img))
    center_coordinates = 0
    box = 0
    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        draw_on_img(file, file_root, a, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1,
                    (max_val, max_val, max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True,
                    label=str(focus_full_row['focus_label']))  # str(focus_full_i)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_procfocusDetectionCheck_Full.tif', a)

    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        draw_on_img(file, file_root, cropped_img, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1,
                    (raw_max_val, raw_max_val, raw_max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True,
                    label=str(focus_full_row['focus_label']))  # str(focus_full_i)
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_rawfocusDetectionCheck_Full.tif', cropped_img.astype('uint16'))

    stop = 1
    print("Done.")
    stop = 1


def pre_process_movies_df(summary_dir, dir_list):
    # dir_list = dir_list[13:19] # for debugging
    """Pre - process - movie df
    Reads from IDR data, and dir_files is the box / folder hosting location of 5th directory in local storage
    Sorting contents within new dir_files and sorted is now label spool_list
    Scans for directories in spool_list with.DS_Store extension and, if it exists, removes from spool_list
    Isolate the first element in spool_list and label spool_list_file df_file create creating pathways to read each
    data set"""
    # block is intended to scan for .DS_Store labelled directories and stores these directories with extension under df_file
    for dir_ in dir_list:
        # Read IDR data
        dir_files = os.listdir(dir_[5])
        spool_list = sorted(dir_files)
        if ('.DS_Store' in spool_list):
            spool_list.remove('.DS_Store')
        spool_list_file = spool_list[0]
        df_file = io.imread(dir_[5] + '/' + spool_list_file)

    # import images
    def last_4chars(x):
        print(x[-8:])
        return (x[-8:])

    # Define directories
    file = spool_list_file  # if only using first frame
    # Find ROI file
    # single ROI
    roi_list = glob(dir_[5] + '/' + spool_list_file[:-4] + '_roi' + '*.zip')
    # check number of rois
    roi_is_zip = True
    rgb_movie = df_file  # np.empty(shape=((len(spool_list)), r_raw, c_raw, 3), dtype='uint16') * 0 #
    for roi_file in roi_list:
        print('Processing ', roi_file)
        # load the ROI and extract the coords
        # make a mask from the coords
        # create folder for this cropped out cell
        # use mask to save the cropped movie file
        """if its zipped file, then the zipped file will be read and stored as roi_points"""
        if (roi_is_zip):
            roi_points = read_roi.read_roi_zip(roi_file)
            """if the zipped file DNE, then it will store the opened roi file that is read in binary mode as 
                fobj and then read fobj and store in roi_points, and then matches file formats as if it is a zip file"""
        else:
            fobj = open(roi_file, 'r+b')
            roi_points = read_roi.read_roi(fobj)
            roi_points = [roi_points, ]  # make format same as if its a zip

    # ROI is bounding box, make mask
    # go through each frame and apply masks, save cropped movie file
    num_frames = len(rgb_movie)
    """roi_count iterates for roi_points, and bbox_points iterates for second element
    bb = bounding box, and enumerate associates incremental value to each roi_point"""
    for roi_count, bbox_points in enumerate(roi_points):  # for debugging roi_points[1:2]
        # roi_count = 1 # for debugging
        print('roi count: ' + str(roi_count))
        if (os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1))):  # delete previous roi directory
            shutil.rmtree(roi_file[:-4] + '_' + str(roi_count + 1))
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1))):  # create roi directory
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1))
        if (not os.path.isdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')):  # create individual foci directory
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci')
        if (not os.path.isdir(roi_file[:-4] + '_' + str(
            roi_count + 1) + '/DetectedFoci_Outlines')):  # create individual foci directory with outlines
            os.mkdir(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines')

    n_rows = bbox_points[2][0] - bbox_points[0][0] + 1
    n_cols = bbox_points[2][1] - bbox_points[0][1] + 1
    file_root = os.path.split(file)[1][:-4]
    # for frame_i in range(num_frames):
    # 0.0 append frames and extract defined rois
    cropped_img = rgb_movie[bbox_points[0][0]:bbox_points[2][0] + 1,
                  bbox_points[0][1]:bbox_points[2][1] + 1]

    # rescale image
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_01_raw_roi' + str(roi_count + 1) + '_beforeResize.tif', cropped_img)

    # resize
    ratio = 2.7  # idr: 10.2 , df: 1.47
    width = int(cropped_img.shape[1] * ratio)
    height = int(cropped_img.shape[0] * ratio)
    dim = (width, height)
    cropped_img = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_LINEAR)
    n_rows_rescale, n_cols_rescale, _ = cropped_img.shape
    n_rows, n_cols, _ = cropped_img.shape

    # 0.1 extract defined roi
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        4] + '_preproc_01_raw_roi' + str(roi_count + 1) + '_afterResize.tif', cropped_img)

    channel_names = ['R', 'G']
    """To loop through all channels, the '1' in the loop should be changed in to 'channel_names' 
        the loop will print the name of each channel it iterates through;
        cv2.createCLAHE is a type of Adaptive Histogram Equalization(AHE) called Contrast Limiting AHE (CLAHE)
        it limits over-contrasting of image to emphasize features of interest"""
    for i_chanl in [1]:  # [0, 1] # to loop through both channels - Remove [1] and replace with channel_names
        # i_chanl = 1 # for debugging
        img = cropped_img[:, :, i_chanl]
        print('image channel: ' + channel_names[i_chanl])
        # 0.2 histogram equalization
        # cliplimit is a threshold value for contrast
        clahe = cv2.createCLAHE(clipLimit=41, tileGridSize=(7, 7))  # 11, (7,7)
        img = clahe.apply(img)
        img_clahe = img.copy()
        #saving newly clipped CLAHE applied image to img and then after copying it saves the
        #enhanced image to local drive
        io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
            4] + '_preproc_02_clahe_roi' + str(roi_count + 1) + '_' + channel_names[i_chanl] + '.tif', img)

        # 0.3 remove hot pixels
        medblurred_img = cv2.medianBlur(img, 3)  # 3
        dif_img = img - medblurred_img
        img = img - dif_img
        # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        # 4] + '_preproc_03_hotpixfilt_roi' + str(roi_count + 1) +'_' + channel_names[i_chanl] + '.tif', img)

        # 0.4 gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)  # (1,1)
        # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        # 4] + '_preproc_04_gaussblur_roi' + str(roi_count + 1) + '_' + channel_names[i_chanl] +'.tif', img)

        # 0.5 background subtraction via k means clustering
        max_pix = np.max(img)
        Z = img.reshape((n_rows_rescale * n_cols_rescale), 1)  # *******************RESHAPE
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3  # 3 # for #1: 4
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        sorted_center = center[:, 0].sort()
        _, img = cv2.threshold(img, int(center[1]), max_pix, cv2.THRESH_TOZERO)
        # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        # 4] + '_preproc_05_bskmeans_roi' + str(roi_count + 1) +'_' + channel_names[i_chanl] + '.tif', img)

        # 1.0 OTSU thresholding
        th = filters.threshold_otsu(img)
        img_mask = img > th
        # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        # 4] + '_proc_11_OTSU_roi' + str(roi_count + 1) +'_' + channel_names[i_chanl] + '.tif', img_mask)

        # 1.1 connected component labeling
        l_, n_ = mh.label(img.reshape(n_rows_rescale, n_cols_rescale),
                          np.ones((3, 3), bool))  # binary_closed_hztl_k #*******************RESCALE
        # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
        # 4] + '_proc_12_ccl_roi' + str(roi_count + 1) +'_' + channel_names[i_chanl] + '.tif', l_)

        # 1.2 measure region properties
        rs_k = regionprops(l_)
        im_props = regionprops(l_,
                               intensity_image=img.reshape(n_rows_rescale, n_cols_rescale))  # *******************RESCALE
        results = []

        if (len(im_props) > 1000):
            continue

    """scanning blobs in im_props folder and labels, stores coordinates, draws rectangle on rois with cv2.boundingRect,
    applies focus tool, crops image"""
    #x:x + w - x is the first coord, x+w is the sum of the width across axis, and this operation calcs width of roi
    #y:y + h - calcs height, reads as y and y+h
    #focus_tot_pix - calcs # of pixels by order of area of roi
    #three-di
    for blob in im_props:
        properties = []
        i_focus = blob.label
        focus_coords = blob.coords
        x, y, w, h = cv2.boundingRect(focus_coords)
        single_focus_img = img[x:x + w, y:y + h]
        single_focus_img_raw = cropped_img[x:x + w, y:y + h, i_chanl]
        rows, cols = single_focus_img_raw.shape
        focus_tot_pix = rows * cols
        (h, w) = single_focus_img_raw.shape
        #If area of pixels (r*c) is less than 3 or greater than 1700, then it will skip to next blob in list of im_props
        #scanning for images between range; 3>pix<1700
        if ((focus_tot_pix < 30) or (focus_tot_pix > 1700)):  # or (blob.area > 1000)
            continue
    # blob = im_props[10] # for debugging
    print('blob label: ' + str(blob.label))
    properties.append(blob.label)
    properties.append(blob.centroid[0])
    properties.append(blob.centroid[1])
    properties.append(blob.orientation)
    properties.append(blob.area)
    properties.append(blob.perimeter)
    properties.append(blob.major_axis_length)
    properties.append(blob.minor_axis_length)
    properties.append(blob.eccentricity)
    properties.append(blob.coords)
    properties.append(blob.bbox)

    proj_img_x = np.mean(single_focus_img, axis=0).reshape(w, )
    proj_img_y = np.mean(single_focus_img, axis=1).reshape(h, )

    l_x = len(proj_img_x)
    l_y = len(proj_img_y)
    if (l_x != l_y):
        if (l_y > l_x):
            diff = l_y - l_x
    proj_img_x = np.pad(proj_img_x, [(0, diff)])

    if (l_x > l_y):
        diff = l_x - l_y
    proj_img_y = np.pad(proj_img_y, [(0, diff)])

    proj_img_x = proj_img_x
    proj_img_y = proj_img_y
    labels_x = np.arange(0, len(proj_img_x), 1)

    fig = plt.figure()
    width = 1
    ax_x = plt.subplot(3, 1, 1)
    ax_x.bar(labels_x, proj_img_x, width, color='g', edgecolor="k")
    ax_x.set_xlabel("")
    ax_x.set_title(' x ')

    labels_y = np.arange(0, len(proj_img_y), 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_y.bar(labels_y, proj_img_y, width, color='g', edgecolor="k")
    ax_y.set_xlabel("")
    ax_y.set_title(' y ')

    ax_xy = plt.subplot(3, 1, 3)
    rects1 = ax_xy.bar(labels_x, proj_img_x, width, color='royalblue', edgecolor="k")
    rects2 = ax_xy.bar(labels_y, proj_img_y, width, color='seagreen', edgecolor="k")
    ax_xy.set_xlabel("")
    ax_xy.legend((rects1[0], rects2[0]), ('x', 'y'))
    plt.close(fig)

    proj_img_x_ = proj_img_x.ravel().astype('float32')
    proj_img_y_ = proj_img_y.ravel().astype('float32')

    compare_val_correl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CORREL)
    compare_val_chisq_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR)
    compare_val_intrsct_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_INTERSECT)
    compare_val_bc_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_BHATTACHARYYA)
    compare_val_chisqalt_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_CHISQR_ALT)
    compare_val_hellinger_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_HELLINGER)
    compare_val_kl_xy = cv2.compareHist(proj_img_x_, proj_img_y_, cv2.HISTCMP_KL_DIV)

    # pixel number
    len_single_focus = single_focus_img.size

    properties.append(compare_val_correl_xy)
    properties.append(compare_val_chisq_xy)
    properties.append(compare_val_intrsct_xy)
    properties.append(compare_val_bc_xy)
    properties.append(compare_val_chisqalt_xy)
    properties.append(compare_val_hellinger_xy)
    properties.append(compare_val_kl_xy)
    properties.append(len_single_focus)
    results.append(properties)

    # save images of individual foci, with and without outlines
    # extract focus bounding box
    focus_bbox = blob.bbox
    img_r1, img_c1 = img_clahe.shape
    # define bounding box
    pixel_border = 3
    r0 = focus_bbox[0] - pixel_border
    r1 = focus_bbox[2] + pixel_border
    c0 = focus_bbox[1] - pixel_border
    c1 = focus_bbox[3] + pixel_border
    # check if bounding box goes beyond image limits
    if (r0 < 0):
        r0 = 0
    if (r1 > img_r1):
        r1 = img_r1
    if (c0 < 0):
        c0 = 0
    if (c1 > img_c1):
        c1 = img_c1
    # focus with no outlines
    img_focus_clahe = img_clahe.copy()[r0:r1, c0:c1]
    # focus with outlines
    # draw outlines on full nucleus ROI
    focus_max_val = int(np.max(img_clahe) / 2)
    obj_mask = np.zeros(shape=img_clahe.shape, dtype='uint8')  # shape=draw_img.shape[:-1]
    """looping through """
    for i in range(len(focus_coords)):
        obj_mask[focus_coords[i][0], focus_coords[i][1]] = 200
    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_clahe, contours, -1, (focus_max_val, focus_max_val, focus_max_val), 1)
    # crop out focus of interest
    img_focus_clahe_outlines = img_clahe[r0:r1, c0:c1]

    # save focus in focus folder
    # max proj
    # no outlines
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci/' + dir_[0] + '_' +
              dir_[1] + '_' + dir_[2] + '_' + dir_[3] + dir_[4] + '_roi' + str(roi_count + 1) +
              '_focusNo' + str(i_focus) + '_' + channel_names[i_chanl] + '.tif', img_focus_clahe)
    # outlines
    io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/DetectedFoci_Outlines/' + dir_[0] + '_' +
              dir_[1] + '_' + dir_[2] + '_' + dir_[3] + dir_[4] + '_roi' + str(roi_count + 1) +
              '_focusNo' + str(i_focus) + '_' + channel_names[i_chanl] + '.tif', img_focus_clahe_outlines)

    df_foci_full = pd.DataFrame(results,
                                columns=['focus_label', 'centroid-0', 'centroid-1', 'orientation', 'area',
                                         'perimeter', 'major_axis_length',
                                         'minor_axis_length', 'eccentricity', 'coords', 'bbox',
                                         'compare_val_correl',
                                         'compare_val_chisq', 'compare_val_intrsct', 'compare_val_bc',
                                         'compare_val_chisqalt', 'compare_val_hellngr', 'compare_val_kl',
                                         'tot_pix_no'])

    # save properties
    df_foci_full.to_csv(
        roi_file[:-4] + '_' + str(roi_count + 1) + '/' + 'focus_info_full_' + channel_names[i_chanl] + '.csv')

    # save labeled and contoured foci
    a = img.copy()
    max_val = int(np.max(a))
    img_1 = np.float32(cropped_img[:, :, i_chanl])
    raw_max_val = int(np.max(img_1))
    center_coordinates = 0
    box = 0
    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        draw_on_img(file, file_root, a, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1,
                    (max_val, max_val, max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True,
                    label=str(focus_full_row['focus_label']))  # str(focus_full_i)
    # io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
    # 4] + '_procfocusDetectionCheck_Full_roi'+ str(roi_count + 1) +'_' + channel_names[i_chanl] + '.tif', a)

    # draw on all detected and processed foci, contours only
    """xyz"""
    for focus_full_i, focus_full_row in df_foci_full.iterrows():
        # print(focus_full_i) # for debugging
        draw_on_img(file, file_root, img_1, focus_full_row['bbox'], focus_full_row['coords'],
                    center_coordinates, box, 1, 1, (raw_max_val, raw_max_val, raw_max_val), draw_bbox=False,
                    draw_contours=True, draw_label=True, label=str(focus_full_row['focus_label']))  # str(focus_full_i)
        io.imsave(roi_file[:-4] + '_' + str(roi_count + 1) + '/' + dir_[0] + '_' + dir_[3] + dir_[
            4] + '_rawfocusDetectionCheck_Full_roi' + str(roi_count + 1) + '_' + channel_names[i_chanl] + '.tif',
                  img_1.astype('uint16'))

        stop = 1

        print("Done.")
        stop = 1

        ########################################################################################################################

        func = 'pre_process'
        base_dir = '/Users/kalkidantadese/desktop/sample_data'  # 'C:/' 'H:/2021_FociData' #'Z:/rothenberglab/archive/Maria/2021_FociData'

        if (func == 'pre_process'):
            dirs_dict = get_dir_list(base_dir)
            for summary_dir in dirs_dict.keys():
                summary_dir = '20200220_1_2'
                pre_process_movies(base_dir + '/' + summary_dir, dirs_dict[summary_dir])

        elif (func == 'pre_process_idr'):
            dirs_dict = get_dir_list(base_dir)
            for summary_dir in dirs_dict.keys():
                summary_dir = '2022_idr_data'
                pre_process_movies_idr(base_dir + '/' + summary_dir, dirs_dict[summary_dir])

        elif (func == 'pre_process_df'):
            dirs_dict = get_dir_list(base_dir)
            for summary_dir in dirs_dict.keys():
                summary_dir = '2022_DF_data'
                pre_process_movies_df(base_dir + '/' + summary_dir, dirs_dict[summary_dir])

"""Similarities
* def function with same parameters (summary_dir, dir_list):
* all loops begin with for dir_ in dir_list:
- aka accessing the same directory 

Pre-process idr and df
* both store sorted list of files in spool_list and save and assign to local variables their read files
* spool_list_file = spool_list[0] - both idr and df assign 1st element from spool_list 
to another local variable called spool_list_file


Differences

Pre-process 1st for loop creates new directories (2-subs /left + /right per directory) and 
adds extension to construct local pathway
- assigns the new paths to a local variable

Pre-process-df
- deletes .DS_store from spool_list, scanning through all files in the sorted directory
- assigned variable is df_file, which is similar to idr_file used in the pre-process-idr function

DRAFT CODE

# combine pre-process movies
   for dir_ in dir_list:
       # Read BP1-2 data
       # dir_ = dir_list[25] # for debugging
       left_chnl_dir = dir_[5] + '/lefts'
       right_chnl_dir = dir_[5] + '/rights'
       left_chnl_dir_files = os.listdir(left_chnl_dir)
       right_chnl_dir_files = os.listdir(right_chnl_dir)
   for dir_ in dir_list:
       # Read IDR data
       dir_files = os.listdir(dir_[5])
       spool_list = sorted(dir_files)
       if ('.DS_Store' in spool_list):
           spool_list.remove('.DS_Store')
       spool_list_file = spool_list[0]
       idr_file = io.imread(dir_[5] + '/' + spool_list_file)
       df_file = io.imread(dir_[5] + '/' + spool_list_file)"""

"""UNDER CONSTRUCTION
# import images
def last_4chars(x):
    print(x[-8:])
    return (x[-8:])

# Compile images; left = red chnl, right, green chnl
# Define directories
spool_list = sorted(left_chnl_dir_files)
single_left_file = io.imread(left_chnl_dir + '/' + spool_list[1])
r_raw, c_raw = single_left_file.shape
spool_list = spool_list[0:3]
# file = spool_list # if only using first frame
# Initialize image
rgb_movie = np.empty(shape=((len(spool_list)), r_raw, c_raw, 3), dtype='uint16') * 0  #
t_ = 0

    r_raw, c_raw = idr_file.shape (how can this line be specific to idr)
file = spool_list_file  # if only using first frame"""