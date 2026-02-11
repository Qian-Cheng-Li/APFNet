from torch.utils.data import Dataset
import numpy as np
import random
import os
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import cv2


# several data augumentation strategies
def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    flip_flag2 = random.randint(0, 1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    if flip_flag2 == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth


def randomCrop(image, label, depth):
    if random.random() > 0.5:
        border = 20
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
            (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return image.crop(random_region), label.crop(random_region), depth.crop(random_region)
    return image, label, depth


def randomRotation(image, label, depth):
    if random.random() > 0.5:
        mode = Image.BICUBIC
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth


def random_noise(image, label, depth, noise_sigma=10):
    if random.random() > 0.5:
        in_hw = image.size[::-1] + (1,)
        noise = np.uint8(np.random.randn(*in_hw) * noise_sigma)  # +-

        image = np.array(image) + noise  # broadcast
        image = Image.fromarray(image, "RGB")
    return image, label, depth


def random_blur(image, label, depth, kernel_size=(3, 3)):
    if random.random() > 0.5:
        assert len(kernel_size) == 2, "kernel size must be tuple and len()=2"
        image = cv2.GaussianBlur(np.array(image), ksize=kernel_size, sigmaX=0)
        image = Image.fromarray(image, "RGB")
    return image, label, depth


# def colorEnhance(image):
#     bright_intensity=random.randint(5,15)/10.0
#     image=ImageEnhance.Brightness(image).enhance(bright_intensity)
#     contrast_intensity=random.randint(5,15)/10.0
#     image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
#     color_intensity=random.randint(0,20)/10.0
#     image=ImageEnhance.Color(image).enhance(color_intensity)
#     sharp_intensity=random.randint(0,30)/10.0
#     image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
#     return image
# def randomGaussian(image, mean=0.1, sigma=0.35):
#     def gaussianNoisy(im, mean=mean, sigma=sigma):
#         for _i in range(len(im)):
#             im[_i] += random.gauss(mean, sigma)
#         return im
#     img = np.asarray(image)
#     width, height = img.shape
#     img = gaussianNoisy(img[:].flatten(), mean, sigma)
#     img = img.reshape([width, height])
#     return Image.fromarray(np.uint8(img))
# def randomPeper(img):
#     img=np.array(img)
#     noiseNum=int(0.0015*img.shape[0]*img.shape[1])
#     for i in range(noiseNum):
#         randX=random.randint(0,img.shape[0]-1)
#         randY=random.randint(0,img.shape[1]-1)
#         if random.randint(0,1)==0:
#             img[randX,randY]=0
#         else:
#             img[randX,randY]=255
#     return Image.fromarray(img)

# def convert_to_edge(mask_np, method):
#     if method == "sobel":
#         grad_x = cv2.Sobel(mask_np, cv2.CV_64F, 1, 0, ksize=3)
#         grad_y = cv2.Sobel(mask_np, cv2.CV_64F, 0, 1, ksize=3)
#         grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
#         _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
#         return binary_output.astype(np.uint8)
#
#     elif method == "laplacian":
#         laplacian = cv2.Laplacian(mask_np, cv2.CV_64F)
#         _, binary_output = cv2.threshold(np.abs(laplacian), 50, 255, cv2.THRESH_BINARY)
#         return binary_output.astype(np.uint8)
#
#     elif method == "canny":
#         edges = cv2.Canny(mask_np, 10, 100)
#         return edges
#
#     elif method == "scharr":
#         grad_x = cv2.Scharr(mask_np, cv2.CV_64F, 1, 0)
#         grad_y = cv2.Scharr(mask_np, cv2.CV_64F, 0, 1)
#         grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
#         _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
#         return binary_output.astype(np.uint8)
#
#     elif method == "prewitt":
#         kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#         kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
#         grad_x = cv2.filter2D(mask_np, cv2.CV_64F, kernel_x)
#         grad_y = cv2.filter2D(mask_np, cv2.CV_64F, kernel_y)
#         grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
#         _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
#         return binary_output.astype(np.uint8)
#
#     else:
#         raise ValueError("Unknown method for edge detection")

def convert_to_edge(mask_np, method, thicken=False):
    if method == "sobel":
        grad_x = cv2.Sobel(mask_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_np, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    elif method == "laplacian":
        laplacian = cv2.Laplacian(mask_np, cv2.CV_64F)
        _, binary_output = cv2.threshold(np.abs(laplacian), 50, 255, cv2.THRESH_BINARY)

    elif method == "canny":
        binary_output = cv2.Canny(mask_np, 10, 100)

    elif method == "scharr":
        grad_x = cv2.Scharr(mask_np, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(mask_np, cv2.CV_64F, 0, 1)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    elif method == "prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = cv2.filter2D(mask_np, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(mask_np, cv2.CV_64F, kernel_y)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        _, binary_output = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError("Unknown method for edge detection")

    if thicken:
        kernel = np.ones((3, 3), np.uint8)
        binary_output = cv2.dilate(binary_output, kernel, iterations=1)

    return binary_output.astype(np.uint8)


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config=False, train=True):
        super(NPY_datasets, self)
        self.status = train
        if train:
            images_list = os.listdir(path_Data + 'train\\images\\')
            masks_list = os.listdir(path_Data + 'train\\masks\\')
            # depths_list = os.listdir(path_Data + 'train\\depths\\')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            # depths_list = sorted(depths_list)
            self.data = []
            # print(len(images_list), len(masks_list), len(points_list))
            for i in range(len(images_list)):
                img_path = path_Data + 'train\\images\\' + images_list[i]
                mask_path = path_Data + 'train\\masks\\' + masks_list[i]
                # depths_path = path_Data + 'train\\depths\\' + depths_list[i]
                self.data.append([img_path, mask_path])
            self.img_transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize([0.20588122, 0.18021357, 0.14168449], [0.12336577, 0.10373283, 0.086231515])])
            self.gt_transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor()])
            self.gt_b_transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor()])
            # # 定义数据增强操作列表
            # self.augmentation_operations = [
            #     cv_random_flip,
            #     randomCrop,
            #     randomRotation,
            #     random_noise,
            #     random_blur
            # ]

        else:
            images_list = os.listdir(path_Data + 'val\\images\\')
            masks_list = os.listdir(path_Data + 'val\\masks\\')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            # print(len(images_list), len(masks_list))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val\\images\\' + images_list[i]
                mask_path = path_Data + 'val\\masks\\' + masks_list[i]
                self.data.append([img_path, mask_path])
            # self.transformer = config.test_transformer
            self.img_transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize([0.20465584, 0.18399397, 0.1484556], [0.12614389, 0.11309516, 0.0980078])])
            self.gt_transform = transforms.Compose([
                # transforms.Resize((config.input_size_h, config.input_size_w)),
                transforms.ToTensor()])

    def __getitem__(self, indx):
        if self.status == True:
            img_path, msk_path = self.data[indx]
            img = Image.open(img_path).convert('RGB')
            msk = Image.open(msk_path).convert('L')
            # PIL → NumPy
            msk_np = np.array(msk)
            # 边缘提取 (选一个方法)
            edge_np = convert_to_edge(msk_np, method='canny', thicken=True)  # or 'sobel', 'laplacian'...
            # NumPy → PIL
            pnt = Image.fromarray(edge_np)
            # pnt = Image.open(pnt_path).convert('L')
            img, msk, pnt = cv_random_flip(img, msk, pnt)
            img, msk, pnt = randomCrop(img, msk, pnt)
            img, msk, pnt = randomRotation(img, msk, pnt)
            img, msk, pnt = random_noise(img, msk, pnt)
            img, msk, pnt = random_blur(img, msk, pnt)

            # # 随机选择一个数据增强操作
            # random_augmentation = random.choice(self.augmentation_operations)
            # img, msk, pnt = random_augmentation(img, msk, pnt)

            img = self.img_transform(img)
            msk = self.gt_transform(msk)
            pnt = self.gt_b_transform(pnt)
            return img, msk, pnt
        else:
            img_path, msk_path = self.data[indx]
            img = Image.open(img_path).convert('RGB')
            msk = Image.open(msk_path).convert('L')
            img = self.img_transform(img)
            name = img_path.split('\\')[-1]
            msk = self.gt_transform(msk)
            return img, msk, name

    def __len__(self):
        return len(self.data)

# class Test_datasets(Dataset):
#     def __init__(self, path_Data, config):
#         super(Test_datasets, self)
#         images_list = os.listdir(path_Data+'val\\images\\')
#         masks_list = os.listdir(path_Data+'val\\masks\\')
#         images_list = sorted(images_list)
#         masks_list = sorted(masks_list)
#         self.data = []
#         for i in range(len(images_list)):
#             img_path = path_Data+'val\\images\\' + images_list[i]
#             mask_path = path_Data+'val\\masks\\' + masks_list[i]
#             self.data.append([img_path, mask_path])
#
#         self.img_transform = transforms.Compose([
#             transforms.Resize((config.input_size_h, config.input_size_w)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.48897059, 0.46548275, 0.4294], [0.22861765, 0.22948039, 0.24054667])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((config.input_size_h, config.input_size_w)),
#             transforms.ToTensor()])
#
#     def __getitem__(self, indx):
#         img_path, msk_path = self.data[indx]
#         img = Image.open(img_path).convert('RGB')
#         msk = Image.open(msk_path).convert('L')
#         img = self.img_transform(img)
#         msk = self.gt_transform(msk)
#         return img, msk
#
#     def __len__(self):
#         return len(self.data)