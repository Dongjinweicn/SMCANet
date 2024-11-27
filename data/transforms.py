import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import PIL 
from PIL import Image
from util.misc import interpolate
from skimage import transform



def Get_foregroud_center(niblabel):
    """
    根据提供的label计算中心坐标
    :param label:The label to derive enter coordinates
    :param label:The num of subjects
    :param 'NumSubject' list with every element:an array representing the foreground center coordinates, of shape[2]
    """
    numpylabel = np.array(niblabel).squeeze()
    # center_numpylabel = numpylabel[:, :, int(numpylabel.shape[2]/2)]
    ##2D
    center_numpylabel = numpylabel[ :, :]
    center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1))
    return center_coord, numpylabel

def Nifity_imageCrop(numpyimage,center_coord,region):
    # center_x, center_y = center_coord
    # shape = numpyimage.shape
    # numpyimagecrop = np.zeros((height, width, shape[2]), dtype=np.float32)
    # numpyimagecrop[0:height, 0:width, :] = \
    #     numpyimage[int(center_x - height/2):int(center_x + height/2),
    #     int(center_y - width / 2):int(center_y + width / 2), :]
    # return numpyimagecrop
    height = region[0]
    depth = region[1]
    #2D
    center_x,center_y = center_coord
    shape = numpyimage.shape
    numpyimagecrop = np.zeros((height,depth), dtype=np.float32)
    x_s = int(center_x - height / 2)
    x_c=int(center_x + height / 2)
    y_s=int(center_y - depth / 2)
    y_c=int(center_y + depth / 2)
    # if shape[1] ==284:
    #     print('x_s,x_c,y_s,y_c,numpyimage.shape',x_s,x_c,y_s,y_c,numpyimage.shape)
    if x_s<0 and y_s>0:
        numpyimagecrop[0:height, 0:depth] = numpyimage[0:height,
                                            int(center_y - depth / 2):int(center_y + depth / 2)]
    elif x_s>0 and y_s<0:
        numpyimagecrop[0:height, 0:depth] = numpyimage[int(center_x - height / 2):int(center_x + height / 2),
                                            0:depth]
    elif x_c < shape[0] and y_c > shape[1]:
        numpyimagecrop[0:height, 0:depth] = numpyimage[int(center_x - height / 2):int(center_x + height / 2),
                                            shape[1]-depth:shape[1]]
    elif x_c > shape[0] and y_c < shape[1]:
        numpyimagecrop[0:height, 0:depth] = numpyimage[shape[0]-depth:shape[0],
                                            int(center_y - depth / 2):int(center_y + depth / 2)]
    else:
        numpyimagecrop[0:height, 0:depth] = numpyimage[int(center_x - height / 2):int(center_x + height / 2),
                                            int(center_y - depth / 2):int(center_y + depth / 2)]

    return numpyimagecrop


def hflip(image, target):
    flipped_image = image
    # if random.random() < 0.5:
    if target['random'][1] < 0.5:
        flipped_image = np.flip(flipped_image,(1))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask,(1))
    # if random.random() < 0.5:
    if target['random'][2] < 0.5:
        flipped_image = np.flip(flipped_image,(2))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask,(2))
    # rotate_choice = int(random.random()*4)
    rotate_choice = int(target['random'][3]*4)
    flipped_image = np.rot90(flipped_image, k=rotate_choice, axes=(1,2))
    if "masks" in target:
        mask = target['masks']
        target['masks'] = np.rot90(mask, k=rotate_choice, axes=(1,2))
    return flipped_image, target

def crop(image, target, region):

    i, j, h, w = region
    cropped_image = image[:, i:i + h, j:j + w]

    # should we do something wrt the original size?
    target["size"] = [h, w]

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        mask = target["masks"]
        target['masks'] = mask[:, i:i + h, j:j + w]

    return cropped_image, target


def pad(images, targets, region):
    image = images.squeeze()
    target = targets
    if "masks" in target:
        mask = target["masks"]
        center_coord, numpylabel = Get_foregroud_center(mask)
        image_crop = Nifity_imageCrop(image, center_coord, region)
        mask_crop = Nifity_imageCrop(numpylabel, center_coord, region)

        # print(image.shape,target["masks"].shape)
        # print(mask.shape)np.expand_dims(rotated_mask, 0)
        mask_crop = np.expand_dims(mask_crop, 0)
        image_crop = np.expand_dims(image_crop, 0)
        target['masks'] = mask_crop

    return image_crop, target


def pad_ori(images, targets, region):

    # assert image.shape == target["masks"].shape
    # image = np.expand_dims(images[0], axis=0)
    image = images[0]

    z,x,y = image.shape
    nx,ny = region
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    #print("source shape:",(z,x,y), "target shape:",(nx,ny))

    if x > nx and y > ny:
        slice_padded = image[:,x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_padded = np.zeros((z,nx, ny), dtype = np.float32)
        if x <= nx and y > ny:
            slice_padded[:, x_c:x_c + x, :] = image[:, :, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_padded[:,:, y_c:y_c + y] = image[:,x_s:x_s + nx, :]
        else:
            slice_padded[:,x_c:x_c + x, y_c:y_c + y] = image[:,:, :]

    # target = np.expand_dims(targets[0], axis=0)
    target = targets[0]
    if "masks" in target:
        mask = target["masks"]
        # print(image.shape,target["masks"].shape)
        # print(mask.shape)
        if x > nx and y > ny:
            mask_padded = mask[:,x_s:x_s + nx, y_s:y_s + ny]
        else:
            mask_padded = np.zeros((z,nx, ny), dtype = np.float32)
            if x <= nx and y > ny:
                mask_padded[:, x_c:x_c + x, :] = mask[:, :, y_s:y_s + ny]
            elif x > nx and y <= ny:
                mask_padded[:,:, y_c:y_c + y] = mask[:,x_s:x_s + nx, :]
            else:
                mask_padded[:,x_c:x_c + x, y_c:y_c + y] = mask[:,:, :]
        
        target['masks'] = mask_padded

    return slice_padded, target 

def resize(image, target, size):
    min_scale = size[0]
    max_scale = size[1]
    img_width = image.shape[1]
    img_height = image.shape[2]
    target_scale = random.uniform(min_scale, max_scale)
    rescaled_width = int(target_scale*img_width)
    rescaled_height = int(target_scale*img_height)
    #random.randint(min_size, max_size)
    rescaled_size = [rescaled_width, rescaled_height]
    image = image.copy()
    image = torch.from_numpy(image)
    rescaled_image = F.resize(image, rescaled_size, interpolation = PIL.Image.NEAREST)
    rescaled_image = rescaled_image.numpy()

    if target is None:
        return rescaled_image, None

    target = target.copy()
    w = rescaled_width
    h = rescaled_height
    target["size"] = torch.tensor([w, h])

    if "masks" in target:
        mask = target['masks']
        # interpolate_mask = mask[:, None].copy()
        interpolate_mask = mask.copy()
        interpolate_mask = torch.from_numpy(interpolate_mask)
        mask = F.resize(interpolate_mask, rescaled_size, interpolation = PIL.Image.NEAREST)
        # mask = interpolate(interpolate_mask.float(), size, mode="nearest")[:, 0] > 0.5
        mask = mask.numpy()
        target['masks'] = mask

    return rescaled_image, target


class Todata(object):
    def __init__(self):
        pass

    def __call__(self, imgs, targets=None):
        img = imgs[0]
        target = targets[0]

        # if "masks" in target:
        #     mask = target['masks']
        #     target['masks'] = mask
        return img,target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        # if random.random() < self.p:
        if target['random'][0] < self.p:
            return hflip(img, target)
        # print('img.shape, target.shape',img.shape, target['masks'].shape)
        return img, target


class RandomResize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)

class Rescale(object):
    def __init__(self):
        pass

    def __call__(self, imgs, targets=None):
        img = imgs[0]
        scale_vector_img = imgs[1]
        target = targets[0]
        scale_vector_target = targets[1]
        img = transform.rescale(img[0,:,:],
                                    scale_vector_img,
                                    order=1,
                                    preserve_range=True,
                                    multichannel=False,
                                    mode='constant')
        img = np.expand_dims(img, axis=0)
        if "masks" in target:
            mask = target['masks']
            mask = transform.rescale(mask[0,:,:],
                                            scale_vector_target,
                                            order=0,
                                            preserve_range=True,
                                            multichannel=False,
                                            mode='constant')
            mask = np.expand_dims(mask, axis=0)
            target['masks'] = mask
        return img,target


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target

class PadOrCropToSize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        crop_height, crop_width = self.size
        padded_img, padded_target = pad(img, target, (crop_height, crop_width))
        return padded_img, padded_target

class RandomRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
    
    @staticmethod
    def get_params(degrees):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle
    def __call__(self, img, target):
        # angle = self.get_params(self.degrees)
        angle = target['angle']
        angle = float(angle.item())
        # print('angleangleangle',angle)
        img = img.copy()
        # print(img.shape,type(img))
        # rotated_img = F.rotate(torch.from_numpy(img), angle, PIL.Image.NEAREST, self.expand, self.center)
        rotated_img = F.rotate(Image.fromarray(np.squeeze(img)), angle, PIL.Image.NEAREST, self.expand, self.center)
        # print(rotated_img.shape,type(rotated_img))
        rotated_img = np.array(rotated_img)
        rotated_img = np.expand_dims(rotated_img,0)
        #  if "masks" in target:
        mask = target['masks']
        mask = mask.copy()
        # rotated_mask = F.rotate(torch.from_numpy(mask), angle, PIL.Image.NEAREST, self.expand, self.center)
        rotated_mask = F.rotate(Image.fromarray(np.squeeze(mask)), angle, PIL.Image.NEAREST, self.expand, self.center)
        rotated_mask =  np.array(rotated_mask)
        rotated_mask = np.expand_dims(rotated_mask, 0)
        target["masks"] = rotated_mask
        # print('rotated_mask.shape,rotated_img.shape',rotated_mask.shape,rotated_img.shape)
        return rotated_img, target

class RandomColorJitter(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        RGB_img = np.repeat(img, 3, axis=0)
        RGB_img = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(torch.from_numpy(RGB_img))
        gray_img = T.Grayscale(num_output_channels=1)(RGB_img)
        return gray_img, target
        

class CenterRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        min_scale = self.size[0]
        max_scale = self.size[1]
        image_width = img.shape[1]
        image_height = img.shape[2]
        if random.random() < 0.7:
            target_scale = random.uniform(min_scale, max_scale)
        else:
            target_scale = 1
        crop_height = int(target_scale*image_height)
        crop_width = int(target_scale*image_width)
        crop_top = max(0, image_height-crop_height)
        crop_left = max(0, image_width-crop_width)
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))

class ToTensor(object):
    def __call__(self, img, target):
        for k, v in target.items():
            if not isinstance(v, str):
                if torch.is_tensor(v) or isinstance(v, (list, tuple)):
                    if torch.is_tensor(v):
                        pass
                    else:
                        target[k] = torch.tensor(v).type(torch.LongTensor)
                else:
                    v = v.copy()
                    target[k] = torch.tensor(v).type(torch.LongTensor)
        if not torch.is_tensor(img):
            img = img.copy()
            img = torch.from_numpy(img)
        return img, target
        # return torch.from_numpy(img), target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if self.mean is None:
            self.mean = image.mean()
        if self.std is None:
            self.std = image.std()
        image = (image - self.mean) / self.std
        if target is None:
            return image, None

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
