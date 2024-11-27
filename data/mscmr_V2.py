import random

import SimpleITK as sitk
import numpy as np
from pathlib import Path

import torch
from torch.utils import data
import nibabel as nib

import data.transforms as T
import os


def make_dir(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)


import medpy.io as medio


def save_run_image(tensor, label):
    imgsavepath = '/home/djw/Pictures/run_image/'
    labsavepath = '/home/djw/Pictures/run_label/'
    make_dir(imgsavepath)
    make_dir(labsavepath)
    image = tensor.cpu().numpy()
    label = label.cpu().numpy()
    label_a = label.astype(np.float32)
    # label_arrage = label_arrage[:, :, np.newaxis]
    # label_arrage = np.concatenate([label_a.transpose(1, 2, 0),label_a.transpose(1, 2, 0),label_a.transpose(1, 2, 0)],axis=2)
    a = str(random.randint(1, 100)) + str(random.randint(1, 100))
    medio.save(image.transpose(1, 2, 0), imgsavepath + a + '.nii.gz')
    medio.save(label_a.transpose(1, 2, 0), labsavepath + a + '.nii.gz')
    # for i in image.shape[0]:
    #     array = image[i,...]
    #     medio.save(array, imgsavepath +str(random.randint(1 , 100) ) +str(i))


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def resamplewithinterpolator(image, interpolator, spacing):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = spacing
    orig_size = np.array(image.GetSize(), dtype=np.int16)
    orig_spacing = list(image.GetSpacing())
    new_spacing[2] = orig_spacing[2]
    # print(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    new_size = [oz * os / nz for oz, os, nz in zip(orig_size, orig_spacing, new_spacing)]
    new_size = np.ceil(new_size).astype(np.int16)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage


class mscmrSeg(data.Dataset):
    def __init__(self, img_folder, lab_folder, lab_values, transforms):
        self._transforms = transforms
        img_paths = list(img_folder.iterdir())
        # img_paths_c0 = list(img_folder.replace('de','c0').iterdir())
        # img_paths_t2 = list((str(img_folder.iterdir())).replace('de','t2'))
        print()
        lab_paths = list(lab_folder.iterdir())
        self.lab_values = lab_values
        self.examples = []
        self.img_dict_de = {}
        self.img_dict_c0 = {}
        self.img_dict_t2 = {}
        self.lab_dict = {}
        # for img_path, img_path_c0,img_path_t2,lab_path in zip(sorted(img_paths), sorted(img_paths_c0),sorted(img_paths_t2),sorted(lab_paths)):
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            # de
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict_de.update({img_name: img})
            # c0
            img_path_c0 = str(img_path).replace('de', 'c0')
            img_c0 = self.read_image(img_path_c0)
            img_name_c0 = img_path_c0.split('/')[-1]
            self.img_dict_c0.update({img_name_c0: img_c0})
            # t2
            img_path_t2 = str(img_path).replace('de', 't2')
            img_t2 = self.read_image(img_path_t2)
            img_name_t2 = img_path_t2.split('/')[-1]
            self.img_dict_t2.update({img_name_t2: img_t2})

            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            print(img_name, img_name_c0, img_name_t2, lab_name)
            self.lab_dict.update({lab_name: lab})
            # self.examples += [(img_name, lab_name, slice, -1, -1) for slice in range(img.shape[0])]
            # assert img.shape[1] == lab.shape[1]
            # self.examples += [(img_name, lab_name, -1, slice, -1) for slice in range(img.shape[1])]
            assert img[0].shape[2] == lab[0].shape[2] == img_t2[0].shape[2] == img_c0[0].shape[2]
            self.examples += [(img_name, img_name_c0, img_name_t2, lab_name, -1, -1, slice) for slice in
                              range(img[0].shape[2])]

    def __getitem__(self, idx):
        img_name_de, img_name_c0, img_name_t2, lab_name, Z, X, Y = self.examples[idx]
        # print('llllll',self.examples)
        # print('ddddd', len(self.examples))
        #
        # print('kkkkk',img_name, lab_name, Z, X, Y)

        if Z != -1:
            img_de = self.img_dict_de[img_name_de][Z, :, :]
            img_c0 = self.img_dict_c0[img_name_c0][Z, :, :]
            img_t2 = self.img_dict_t2[img_name_t2][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img_de = self.img_dict_de[img_name_de][:, X, :]
            img_c0 = self.img_dict_c0[img_name_c0][:, X, :]
            img_t2 = self.img_dict_t2[img_name_t2][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]

        elif Y != -1:
            # print('self.img_dict', len(self.img_dict))
            img_de = self.img_dict_de[img_name_de][0][:, :, Y]
            scale_vector_img = self.img_dict_de[img_name_de][1]
            img_c0 = self.img_dict_c0[img_name_c0][0][:, :, Y]
            scale_vector_img_c0 = self.img_dict_c0[img_name_c0][1]
            img_t2 = self.img_dict_t2[img_name_t2][0][:, :, Y]
            scale_vector_img_t2 = self.img_dict_t2[img_name_t2][1]

            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        img_de = np.expand_dims(img_de, 0)
        img_c0 = np.expand_dims(img_c0, 0)
        img_t2 = np.expand_dims(img_t2, 0)
        lab = np.expand_dims(lab, 0)

        img_de = img_de.astype(np.float32)
        img_c0 = img_c0.astype(np.float32)
        img_t2 = img_t2.astype(np.float32)
        lab_de = lab.astype(np.float32)
        # print('img.shape,target.shape', img_de.shape, lab_de.shape)

        aa, bb, cc, dd = random.random(), random.random(), random.random(), random.random()
        degrees = (0, 360)
        angle = torch.empty(1).uniform_(float(degrees[0]), float(degrees[1]))
        target_de = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab_de, 'orig_size': lab.shape,
                     'random': [aa, bb, cc, dd], 'angle': angle}
        target_c0 = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab_de, 'orig_size': lab.shape,
                     'random': [aa, bb, cc, dd], 'angle': angle}
        target_t2 = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab_de, 'orig_size': lab.shape,
                     'random': [aa, bb, cc, dd], 'angle': angle}
        # print('00000',target_de["masks"].shape,target_c0["masks"].shape,target_t2["masks"].shape)

        if self._transforms is not None:
            # TODO signal slice transforms, not 3D volume transforms
            img_de, target_de = self._transforms([img_de, scale_vector_img], [target_de, scale_vector_lab])
            img_c0, target_c0 = self._transforms([img_c0, scale_vector_img_c0], [target_c0, scale_vector_lab])
            img_t2, target_t2 = self._transforms([img_t2, scale_vector_img_t2], [target_t2, scale_vector_lab])

        # print('target_de.max()', target_de['masks'].max())
        # save_run_image(torch.cat([img_c0,img_de,img_t2],dim=0),torch.cat([target_c0['masks'],target_de['masks'],target_t2['masks']],dim=0))
        # save_run_image(image_tree,target_de['masks'])

        # image_list = [img_de, img_c0, img_t2]
        # random.shuffle(image_list)
        # return torch.cat(image_list, dim=0), target_de
        return torch.cat([img_c0, img_de, img_t2], dim=0), target_de

    def read_image(self, img_path):
        img_dat = sitk.ReadImage(img_path)
        # spacing = [0.5, 0.5, img_dat.GetSpacing()[2]]
        spacing = [0.72906, 0.72906, img_dat.GetSpacing()[2]]
        new_image = resamplewithinterpolator(img_dat, sitk.sitkLinear, spacing)
        img = sitk.GetArrayFromImage(new_image).transpose(1, 2, 0)
        scale_vector = (None, None)
        img = img.astype(np.float32)
        return [(img - img.mean()) / img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = sitk.ReadImage(lab_path)
        # spacing = [0.5, 0.5, lab_dat.GetSpacing()[2]]
        spacing = [0.72906, 0.72906, lab_dat.GetSpacing()[2]]
        new_label = resamplewithinterpolator(lab_dat, sitk.sitkNearestNeighbor, spacing)
        lab = sitk.GetArrayFromImage(new_label).transpose(1, 2, 0)
        # scale_vector = (1.32812, 1.32812)
        scale_vector = (None, None)
        # lab = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        # lab = np.sum(lab, 0)
        return [lab, scale_vector]

    #
    # def read_image(self, img_path):
    #     img_dat = load_nii(img_path)
    #     img = img_dat[0]
    #     pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
    #     # print(pixel_size)
    #     target_resolution = (1., 1.)
    #     scale_vector = (pixel_size[0] / target_resolution[0],
    #                     pixel_size[1] / target_resolution[1])
    #     img = img.astype(np.float32)
    #     return [(img-img.mean())/img.std(), scale_vector]
    #
    #
    # def read_label(self, lab_path):
    #     lab_dat = load_nii(lab_path)
    #     lab = lab_dat[0]
    #     pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
    #     # target_resolution = (1.32812, 1.32812)
    #     target_resolution = (1., 1.)
    #     scale_vector = (pixel_size[0] / target_resolution[0],
    #                     pixel_size[1] / target_resolution[1])
    #     # lab = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
    #     # lab = np.sum(lab, 0)
    #     return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        # return T.Compose([T.RandomHorizontalFlip(),T.RandomCrop([256, 256]), T.ToTensor()])
        return T.Compose([
            # T.Rescale(),
            T.Todata(),
            T.RandomHorizontalFlip(),
            # T.RandomRotate((0,360)),
            # T.CenterRandomCrop([0.7,1]),
            # T.RandomResize([0.8,1.2]),
            # T.PadOrCropToSize([212,212]),
            T.PadOrCropToSize([196, 196]),  # xiugai 0505
            # T.RandomColorJitter(),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.Todata(),
            # T.RandomHorizontalFlip(),
            # T.CenterRandomCrop([0.7,1]),
            # T.RandomResize([0.8,1.2]),
            # T.Rescale(),
            # T.PadOrCropToSize([212,212]),
            T.PadOrCropToSize([196, 196]),  # xiugai 0505
            # T.RandomColorJitter(),
            normalize])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    # root = Path('/home/gaoshangqi/Segmentation/Datasets/' + args.dataset)
    # root = Path('/media/dong/M/data/MSCMRseg/' + args.dataset)
    # root = Path('/media/dong/M/data/MSCMRseg/' + bayesian_ZS_run)
    # root = Path('/media/Data/djw/data/' + args.dataset)
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images_de", root / "train" / "labels_scar"),
        "val": (root / "val" / "images_de", root / "val" / "labels_scar"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg(img_task, lab_task, lab_values, transforms=make_transforms(image_set))
        dataset_dict.update({task: dataset})
    return dataset_dict

