import torchvision.transforms.functional as F
import torch.nn.functional as Func
import torchvision.transforms as T
import math
import sys
import random
import time
import datetime
import tqdm
from typing import Iterable
import numpy as np
import PIL
from PIL import Image
from skimage import transform
import nibabel as nib
import torch
import os
from medpy.metric.binary import dc
import pandas as pd
import glob
import re
import shutil
import copy
from skimage import measure
from data.mscmr_V2 import resamplewithinterpolator, save_run_image

import util.misc as utils


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

def sort_glob(dir):
    files=glob.glob(dir)
    files.sort()
    return files

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def convert_targets(targets, device):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks

def conv_int(i):
    return int(i) if i.isdigit() else i

def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    # keep a heart connectivity 
    mask_shape = mask.shape

    heart_slice = np.where((mask>0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id

    # keep LV/RV/MYO connectivity
    # out_img = np.zeros(mask.shape, dtype=np.uint8)
    # for struc_id in [1, 2, 3]:
    #     binary_img = out_heart == struc_id
    #     blobs = measure.label(binary_img, connectivity=1)
    #     props = measure.regionprops(blobs)
    #     if not props:
    #         continue
    #     area = [ele.area for ele in props]
    #     largest_blob_ind = np.argmax(area)
    #     largest_blob_label = props[largest_blob_ind].label
    #     out_img[blobs == largest_blob_label] = struc_id
    #final_img = out_img
    final_img = out_heart * mask
    return final_img

def data_load_zscore(img_path,target_resolution):
    img_dat = load_nii(img_path)
    img = img_dat[0].copy()

    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
    scale_vector = (pixel_size[0] / target_resolution[0],
                    pixel_size[1] / target_resolution[1])

    img = img.astype(np.float32)
    img = np.divide((img - np.mean(img)), np.std(img))

    # print(img.shape, pixel_size)
    slice_rescaleds = []
    for slice_index in range(img.shape[2]):
        img_slice = np.squeeze(img[:, :, slice_index])
        slice_rescaled = transform.rescale(img_slice,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           multichannel=False,
                                           anti_aliasing=True,
                                           mode='constant')
        slice_rescaleds.append(slice_rescaled)
    img = np.stack(slice_rescaleds, axis=2)
    return img, img_dat


def resize_crop(img,slice_index,device):
    img_slice = img[:, :, slice_index]
    # nx = 212
    # ny = 212
    nx = 196 # xiugai0505
    ny = 196
    x, y = img_slice.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    # Crop section of image for prediction
    if x > nx and y > ny:
        slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

    # img_slice = slice_cropped
    img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
    img_slice = np.reshape(img_slice, (1, 1, nx, ny))

    img_slice = torch.from_numpy(img_slice)
    img_slice = img_slice.to(device)
    img_slice = img_slice.float()
    return img_slice


@torch.no_grad()
def infer(model, model_type, dataset, sequence, dataloader_dict, output_foldera, device):
    model.eval()
    #criterion.eval()

    #dataset = 'MSCMR'
    # if dataset in ['MSCMR', 'ACDC']:
    if dataset in ['MSCMR']:
        test_files = sort_glob("{}/*/*LGE.nii.gz".format(dataset))
        # label_folder = "/media/Data/djw/data/{}/test/{}/labels_scar/".format(dataset, sequence)
    elif dataset in ['CARE2024_MyoPS']:
        test_files = sort_glob("{}/*/*LGE.nii.gz".format(dataset))
    elif dataset in ['./input']:
        test_files = sort_glob("{}/*/*LGE.nii.gz".format(dataset))
        # label_folder = "/media/Data/djw/data/{}/test/{}/labels_scar/".format(dataset, sequence)
    else:
        raise ValueError('Invalid dataset: {}'.format(dataset))



    output_folder = os.path.join("./", 'output')
    # if os.path.exists(output_folder):
    #     shutil.rmtree(output_folder)
    makefolder(output_folder)

    # target_resolution = (1., 1.)
    target_resolution = (0.72906, 0.72906)

    # test_files = sorted(os.listdir(test_folder))
    # label_files = sorted(os.listdir(label_folder))
    # assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index]
        print('Processing {}'.format(test_file))

        predictions = []

        # label_file = label_files[file_index]
        # file_mask = os.path.join(label_folder, label_file)
        # mask_dat = load_nii(file_mask)
        # mask = mask_dat[0]

        img_path_de = test_file
        img_path_c0 = img_path_de.replace('LGE','C0')
        img_path_t2 = img_path_de.replace('LGE','T2')

        img_de, img_dat = data_load_zscore(img_path_de, target_resolution)
        img_c0,_ = data_load_zscore(img_path_c0, target_resolution)
        img_t2,_ = data_load_zscore(img_path_t2, target_resolution)


        for slice_index in range(img_de.shape[2]):
            img_s = img_de[:, :, slice_index]
            # nx = 212
            # ny = 212
            nx = 196  # xiugai 0505
            ny = 196
            x, y = img_s.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            img_slice_de = resize_crop(img_de,slice_index,device)
            img_slice_c0 = resize_crop(img_c0,slice_index,device)
            img_slice_t2 = resize_crop(img_t2,slice_index,device)
            # print(img_slice_de.shape)
            # img_slice = np.concatenate([img_slice_de,img_slice_c0,img_slice_t2],axis=1)

            # image_list = [img_slice_de,img_slice_c0,img_slice_t2]
            # random.shuffle(image_list)
            # img_slice = torch.cat(image_list,dim=1)
            img_slice = torch.cat([img_slice_c0,img_slice_de,img_slice_t2],dim=1)
            # save_run_image(img_slice.squeeze(), img_slice.squeeze())

            # tasks = dataloader_dict.keys()
            # task = random.sample(tasks, 1)[0]
            task = 'MR'

            if model_type in ['Unet', 'Baseline','SMCAnet']:
                outputs = model(img_slice, task)
            elif model_type == 'PUnet':
                outputs = model(img_slice, None, training=False)
            else:
                return ValueError('Invalid model: {}'.format(model_type))

            # softmax_out = torch.cat([outputs["pred_masks_q"],outputs["pred_masks_k"]], dim=0)  # xiugai0718
            softmax_out = outputs["pred_masks_q"]

            # softmax_out = outputs["pred_masks"]

            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0,...])

            num_class = 6
            slice_predictions = np.zeros((num_class,x,y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:,x_s:x_s+nx, y_s:y_s+ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:,:, y_s:y_s+ny] = prediction_cropped[:,x_c:x_c+ x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:,x_s:x_s + nx, :] = prediction_cropped[:,:, y_c:y_c + y]
                else:
                    slice_predictions[:,:, :] = prediction_cropped[:,x_c:x_c+ x, y_c:y_c + y]
            # print('slice_predictions',slice_predictions.shape,slice_predictions.max())
            prediction = transform.resize(slice_predictions,
                                (num_class, img_dat[0].shape[0], img_dat[0].shape[1]),
                                order=1,
                                preserve_range=True,
                                anti_aliasing=True,
                                mode='constant')
            # print('prediction',prediction.shape,prediction.max())

            prediction = np.uint8(np.argmax(prediction, axis=0))
            #prediction = keep_largest_connected_components(prediction)
            # print('prediction2222222',prediction.shape,prediction.max())
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

        prediction_arr = np.where(prediction_arr == 1, 0, prediction_arr)
        prediction_arr = np.where(prediction_arr == 2, 0, prediction_arr)
        prediction_arr = np.where(prediction_arr == 3, 0, prediction_arr)
        prediction_arr = np.where(prediction_arr == 4, 2221, prediction_arr)
        prediction_arr = np.where(prediction_arr == 5, 1220, prediction_arr)

        # dir_pred = output_folder + test_file.split('/')
        # makefolder(dir_pred)
        out_file_name = output_folder +'/'+ test_file.split('/')[-2] + '_pred.nii.gz'
        out_affine = img_dat[1]
        out_header = img_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)
