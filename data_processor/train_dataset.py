"""
@author: Yuxi Liu
@date: 20230401
@contact: liuyuxi.tongji@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import copy
'''
landmarks = np.array([[36.2946, 51.5014],
                      [76.5318, 51.5014],
                      [56.0252, 71.7366],
                      [41.5493, 92.2041],
                      [70.7299, 92.2041]
                      ], dtype=np.float32 )
'''
'''
landmarks = np.array([[30.2946, 51.6963],
                      [65.5318, 51.5014],
                      [48.0252, 71.7366],
                      [33.5493, 92.3655],
                      [62.7299, 92.2041]
                      ], dtype=np.float32 )
'''
landmarks = np.array([[45.0, 44.0],
                      [67.0, 44.0],
                      [56.0, 62.0],
                      [41.0, 78.0],
                      [71.0, 78.0]
                      ], dtype=np.float32 )
def transform(image):
    """ Transform a image by cv2.
    """
    image=cv2.resize(image, (112, 112))
    img_size = image.shape[0]
    # random crop
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.8:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image


# class ImageDataset(Dataset):
#     def __init__(self, data_root, train_file, crop_eye=False):
#         self.data_root = data_root
#         self.train_list = []
#         train_file_buf = open(train_file)
#         line = train_file_buf.readline().strip()
#         while line:
#             image_path, image_label = line.split(' ')
#             self.train_list.append((image_path, int(image_label)))
#             line = train_file_buf.readline().strip()
#         self.crop_eye = crop_eye
#     def __len__(self):
#         return len(self.train_list)
#     def __getitem__(self, index):
#         image_path, image_label = self.train_list[index]
#         image_path = os.path.join(self.data_root, image_path)
#         image = cv2.imread(image_path)
#         if self.crop_eye:
#             image = image[:60, :]
#         #image = cv2.resize(image, (128, 128)) #128 * 128
#         if random.random() > 0.5:
#             image = cv2.flip(image, 1)
#         if image.ndim == 2:
#             image = image[:, :, np.newaxis]
#         image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
#         image = torch.from_numpy(image.astype(np.float32))
#         return image, image_label

class ImageDataset_Crop(Dataset):
    def __init__(self, data_root, train_file, crop_eye_or_mouth=True,transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                image_path, label = line.strip().split(' ')
                self.train_list.append((image_path, int(label)))
        self.crop_eye_or_mouth = crop_eye_or_mouth
        self.transform = transform       
        self.preprocess = preprocess

    def __len__(self):
        return len(self.train_list)
    
    def random_crop_eye_or_mouth(self, img, landmarks, padding=10):
        choice = random.choice(['eye', 'mouth'])
        if choice == 'eye':
            points = np.array([landmarks[0], landmarks[1]])  # 两只眼睛
        else:
            points = np.array([landmarks[3], landmarks[4]])  # 嘴角

        min_x = int(np.min(points[:, 0]).item())
        max_x = int(np.max(points[:, 0]).item())
        min_y = int(np.min(points[:, 1]).item())
        max_y = int(np.max(points[:, 1]).item())

        width = max_x - min_x
        height = max_y - min_y

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        new_width = int(width * 0.8)
        new_height = int(height * 0.4)

        min_x = center_x - new_width // 2
        max_x = center_x + new_width // 2
        min_y = center_y - new_height // 2
        max_y = center_y + new_height // 2

        min_x = max(0, min_x - padding)
        max_x = min(img.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(img.shape[0], max_y + padding)

        img[min_y:max_y, min_x:max_x] = (255, 255, 255)

        return img

    
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        full_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {full_path}")

        # 随机裁剪眼睛或嘴巴
        if self.crop_eye_or_mouth:
            image_cropped = self.random_crop_eye_or_mouth(image.copy(), landmarks)
        else:
            image_cropped = image.copy()


        # resize 到 112x112 保持大小一致
        image_cropped = cv2.resize(image_cropped, (112, 112))
        image = cv2.resize(image, (112, 112))

        # transform 和 preprocess
        if self.transform is not None:
            image_cropped_t = self.transform(image_cropped)
            image_t = self.transform(image)
        else:
            image_cropped_t = torch.from_numpy(image_cropped)
            image_t = torch.from_numpy(image)

        if self.preprocess is not None:
            image_cropped_clip = self.preprocess(Image.fromarray(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)))
            image_clip = self.preprocess(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        else:
            image_cropped_clip = None
            image_clip = None

        # cgs表示类别，这里不做遮挡，固定为0
        cgs = 0

        return image_cropped_t, image_cropped_clip, image_t, image_label, cgs




def random_crop_eye_or_mouth1(image, crop_height=60):
    height = image.shape[0]
    if height < crop_height:
        return image
    if random.random() < 0.5:
        return image[:crop_height, :, :]
    else:
        return image[-crop_height:, :, :]

class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, crop_part=False, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))
        self.crop_part = crop_part
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        image_path, label = self.train_list[index]
        full_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {full_path}")

        # 这里要定义或传入landmarks，先用固定值演示
        landmarks = np.array([[45.0, 44.0],
                          [67.0, 44.0],
                          [56.0, 62.0],
                          [41.0, 78.0],
                          [71.0, 78.0]], dtype=np.float32)

        if self.crop_eye_or_mouth:
            image_cropped = self.random_crop_eye_or_mouth(image.copy(), landmarks)
        else:
            image_cropped = image.copy()


        image_cropped = cv2.resize(image_cropped, (112, 112))
        image = cv2.resize(image, (112, 112))

        # 转成Tensor，归一化到[0,1]
        image_cropped_t = torch.from_numpy(image_cropped.transpose(2, 0, 1)).float() / 255.0
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 这里暂时没clip预处理，返回None
        image_cropped_clip = None
        image_clip = None

        cgs = 0  # 你训练代码里用不到可以先固定0

        return image_cropped_t, image_cropped_clip, image_t, label, cgs

class ImageDataset_SST(Dataset):
    def __init__(self, data_root, train_file, exclude_id_set):
        self.data_root = data_root
        label_set = set()
        # get id2image_path_list
        self.id2image_path_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, label = line.split(' ')
            label = int(label)
            if label in exclude_id_set:
                line = train_file_buf.readline().strip()
                continue
            label_set.add(label)
            if not label in self.id2image_path_list:
                self.id2image_path_list[label] = []
            self.id2image_path_list[label].append(image_path)
            line = train_file_buf.readline().strip()
        self.train_list = list(label_set)
        print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_list = self.id2image_path_list[cur_id]
        if len(cur_image_path_list) == 1:
            image_path1 = cur_image_path_list[0]
            image_path2 = cur_image_path_list[0]
        else:
            training_samples = random.sample(cur_image_path_list, 2)
            image_path1 = training_samples[0]
            image_path2 = training_samples[1]
        image_path1 = os.path.join(self.data_root, image_path1)
        image_path2 = os.path.join(self.data_root, image_path2)
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image1 = transform(image1)
        image2 = transform(image2)
        if random.random() > 0.5:
            return image2, image1, cur_id
        return image1, image2, cur_id


class ImageDataset_KD(Dataset):
    def __init__(self, data_root, train_file, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))

        self.transform = transform
        self.preprocess = preprocess

       # 加载 mask/glasses/sunglasses 图片
        base_path = "/home/srp/face_recognition/training_mode/conventional_training"
        self.mask_img = cv2.imread(os.path.join(base_path, "mask_img.png"), cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread(os.path.join(base_path, "glass_img3.png"), cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread(os.path.join(base_path, "sunglass_img2.png"), cv2.IMREAD_UNCHANGED)

        # 检查是否加载成功
        assert self.mask_img is not None, "mask_img.png 加载失败，请检查路径"
        assert self.glass_img is not None, "glass_img3.png 加载失败，请检查路径"
        assert self.sunglass_img is not None, "sunglass_img2.png 加载失败，请检查路径"

    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        prob=random.uniform(0, 1)
        if (prob<0.33):
            masked_sample=self.mask_images(sample)
            # plt.imshow(masked_sample.astype('uint8'))
            # 保存图片
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
             sample = self.transform(sample)
             masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
             masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 1
        elif (prob<0.66):
            masked_sample=self.mask_images_sunglass(sample)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 2
        else:
            if self.transform is not None:
                sample_t = self.transform(sample)
            if self.preprocess is not None:
             sample_clip=self.preprocess(Image.fromarray(np.uint8(sample)))
            return sample_t,sample_clip,sample_t,image_label, 0
    
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_glass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_sunglass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.sunglass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg

class ImageDataset_KD_glasses(Dataset):
    def __init__(self, data_root, train_file, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))

        self.transform = transform
        self.preprocess = preprocess

        # ✅ 加载 mask/glasses/sunglasses 图片
        base_path = "/home/srp/face_recognition/training_mode/conventional_training"
        self.mask_img = cv2.imread(os.path.join(base_path, "mask_img.png"), cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread(os.path.join(base_path, "glass_img3.png"), cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread(os.path.join(base_path, "sunglass_img2.png"), cv2.IMREAD_UNCHANGED)

        # ✅ 检查是否加载成功
        assert self.mask_img is not None, "❌ mask_img.png 加载失败，请检查路径"
        assert self.glass_img is not None, "❌ glass_img3.png 加载失败，请检查路径"
        assert self.sunglass_img is not None, "❌ sunglass_img2.png 加载失败，请检查路径"

    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        prob=random.uniform(0, 1)
        if (prob<0.33):
            masked_sample=self.mask_images(sample)
            # plt.imshow(masked_sample.astype('uint8'))
            # 保存图片
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
             sample = self.transform(sample)
             masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
             masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 1
        elif (prob<0.66):
            masked_sample=self.mask_images_glass(sample)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 2
        else:
            if self.transform is not None:
                sample_t = self.transform(sample)
            if self.preprocess is not None:
             sample_clip=self.preprocess(Image.fromarray(np.uint8(sample)))
            return sample_t,sample_clip,sample_t,image_label, 0
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img
    def mask_images_glass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg

class ImageDataset_KD_glasses_sunglasses(Dataset):
    def __init__(self, data_root, train_file, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))

        self.transform = transform
        self.preprocess = preprocess

        # ✅ 加载 mask/glasses/sunglasses 图片
        base_path = "/home/srp/face_recognition/training_mode/conventional_training"
        self.mask_img = cv2.imread(os.path.join(base_path, "mask_img.png"), cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread(os.path.join(base_path, "glass_img3.png"), cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread(os.path.join(base_path, "sunglass_img2.png"), cv2.IMREAD_UNCHANGED)

        # ✅ 检查是否加载成功
        assert self.mask_img is not None, "❌ mask_img.png 加载失败，请检查路径"
        assert self.glass_img is not None, "❌ glass_img3.png 加载失败，请检查路径"
        assert self.sunglass_img is not None, "❌ sunglass_img2.png 加载失败，请检查路径"

    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        prob=random.uniform(0, 1)
        # resize_prob=random.uniform(0, 0.25)
        masked_sample_clip=sample
        sample_clip=sample
        if (prob<0.25):
            masked_sample=self.mask_images(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # plt.imshow(masked_sample.astype('uint8'))
            # 保存图片
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
             sample = self.transform(sample)
             masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
             masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 1
        elif (prob<0.5):
            masked_sample=self.mask_images_glass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 2
        elif (prob<0.75):
            masked_sample=self.mask_images_sunglass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 3
        else:
            # sample=self.crop_images(sample,resize_prob)
            if self.transform is not None:
                sample_t = self.transform(sample)
            if self.preprocess is not None:
             sample_clip=self.preprocess(Image.fromarray(np.uint8(sample)))
            return sample_t,sample_clip,sample_t,image_label, 0
    def crop_images(self,img,prob):
        if(prob>0.5):
            return img
        else:
            # print('size',img.shape)
            new_size=int((img.shape[0]*(1+prob))//2*2)
            # print(new_size)
            new_img=img
            new_img=cv2.resize(img, (new_size, new_size),interpolation=cv2.INTER_LINEAR)
            begin=(new_size-img.shape[0])//2
            end=begin+img.shape[0]
            new_img=new_img[begin:end,begin:end,:]
            return new_img
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_glass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)

        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_sunglass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.sunglass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg

class ImageDataset_HSST(Dataset):
    def __init__(self, data_root,data_mask_root, train_file, exclude_id_set):
        self.data_root = data_root
        self.data_mask_root = data_mask_root
        label_set = set()
        # get id2image_path_nir_list & id2image_path_vis_list
        self.id2image_path_nir_list = {}
        self.id2image_path_vis_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            vis_img_path, label = line.split(' ')
            nir_img_path = vis_img_path
            '''
            path_list = nir_img_path.split('/')
            path_list[-3]+='_random_mask'
            nir_img_path = '/'
            nir_img_path = nir_img_path+'/'+path_liss_one for path_liss_one in path_list
            print(nir_img_path)
            '''
            label = int(label)
            if label in exclude_id_set:
                line = train_file_buf.readline().strip()
                continue
            label_set.add(label)
            if not label in self.id2image_path_nir_list:
                self.id2image_path_nir_list[label] = []
            if not label in self.id2image_path_vis_list:
                self.id2image_path_vis_list[label] = []
            self.id2image_path_nir_list[label].append(nir_img_path)
            self.id2image_path_vis_list[label].append(vis_img_path)
            line = train_file_buf.readline().strip()

        self.train_list = list(label_set)
        # print(self.train_list,self.id2image_path_nir_list,self.id2image_path_vis_list)
        print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_nir_list = list(set(self.id2image_path_nir_list[cur_id]))
        cur_image_path_vis_list = list(set(self.id2image_path_vis_list[cur_id]))
        nir_path = random.sample(cur_image_path_nir_list, 1)[0]
        vis_path = random.sample(cur_image_path_vis_list, 1)[0]
        nir_path = os.path.join(self.data_mask_root, nir_path)
        vis_path = os.path.join(self.data_root, vis_path)
        nir_image = cv2.imread(nir_path)
        vis_image = cv2.imread(vis_path)
        nir_image = transform(nir_image)
        vis_image = transform(vis_image)

        return nir_image, vis_image, cur_id
class ImageDataset_KD_glasses_sunglasses_one(Dataset):
    def __init__(self, data_root, train_file, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))

        self.transform = transform
        self.preprocess = preprocess

        # ✅ 加载 mask/glasses/sunglasses 图片
        base_path = "/home/srp/face_recognition/training_mode/conventional_training"
        self.mask_img = cv2.imread(os.path.join(base_path, "mask_img.png"), cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread(os.path.join(base_path, "glass_img3.png"), cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread(os.path.join(base_path, "sunglass_img2.png"), cv2.IMREAD_UNCHANGED)

        # ✅ 检查是否加载成功
        assert self.mask_img is not None, "❌ mask_img.png 加载失败，请检查路径"
        assert self.glass_img is not None, "❌ glass_img3.png 加载失败，请检查路径"
        assert self.sunglass_img is not None, "❌ sunglass_img2.png 加载失败，请检查路径"

    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=image
        prob=random.uniform(0, 1)
        # resize_prob=random.uniform(0, 0.25)
        masked_sample_clip=sample
        sample_clip=sample
        if (prob<=0.5):
            mask_type=self.mask_type
        else:
            mask_type=None
        if (mask_type=='mask'):
            masked_sample=self.mask_images(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # plt.imshow(masked_sample.astype('uint8'))
            # 保存图片
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
             sample = self.transform(sample)
             masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
             masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            # print('mask')
            return masked_sample_t, masked_sample_clip, sample, image_label, 1
        elif (mask_type=='glasses'):
            masked_sample=self.mask_images_glass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 2
        elif (mask_type=='sunglasses'):
            masked_sample=self.mask_images_sunglass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            # img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            # img.save('./out_pic_sunglasses/b_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            if self.transform is not None:
                sample = self.transform(sample)
                masked_sample_t=self.transform(masked_sample)
            if self.preprocess is not None:
                masked_sample_clip=self.preprocess(Image.fromarray(np.uint8(masked_sample)))
            return masked_sample_t, masked_sample_clip, sample, image_label, 3
        else:
            # sample=self.crop_images(sample,resize_prob)
            if self.transform is not None:
                sample_t = self.transform(sample)
            if self.preprocess is not None:
             sample_clip=self.preprocess(Image.fromarray(np.uint8(sample)))
            # print('no mask')
            return sample_t,sample_clip,sample_t,image_label, 0
    def crop_images(self,img,prob):
        if(prob>0.5):
            return img
        else:
            # print('size',img.shape)
            new_size=int((img.shape[0]*(1+prob))//2*2)
            # print(new_size)
            new_img=img
            new_img=cv2.resize(img, (new_size, new_size),interpolation=cv2.INTER_LINEAR)
            begin=(new_size-img.shape[0])//2
            end=begin+img.shape[0]
            new_img=new_img[begin:end,begin:end,:]
            return new_img
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_glass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_sunglass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.sunglass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg
        
class ImageDataset_KD_glasses_sunglasses_save(Dataset):
    def __init__(self, data_root, train_file, transform=None, preprocess=None):
        self.data_root = data_root
        self.train_list = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, image_label = line.split(' ')
                self.train_list.append((image_path, int(image_label)))

        self.transform = transform
        self.preprocess = preprocess

        # ✅ 加载 mask/glasses/sunglasses 图片
        base_path = "/home/srp/face_recognition/training_mode/conventional_training"
        self.mask_img = cv2.imread(os.path.join(base_path, "mask_img.png"), cv2.IMREAD_UNCHANGED)
        self.glass_img = cv2.imread(os.path.join(base_path, "glass_img3.png"), cv2.IMREAD_UNCHANGED)
        self.sunglass_img = cv2.imread(os.path.join(base_path, "sunglass_img2.png"), cv2.IMREAD_UNCHANGED)

        # ✅ 检查是否加载成功
        assert self.mask_img is not None, "❌ mask_img.png 加载失败，请检查路径"
        assert self.glass_img is not None, "❌ glass_img3.png 加载失败，请检查路径"
        assert self.sunglass_img is not None, "❌ sunglass_img2.png 加载失败，请检查路径"
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112)) #128 * 128
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        # return image, image_label
        sample=copy.deepcopy(image)
        prob=random.uniform(0, 1)
        # resize_prob=random.uniform(0, 0.25)
        masked_sample_clip=sample
        sample_clip=sample
        if (1):
            sample=copy.deepcopy(image)
            masked_sample=self.mask_images(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save('./out_pic_case/maskimg%d.jpg'%index)
            # print('mask')
            # return masked_sample_t, masked_sample_clip, sample, image_label, 1
        if (1):
            sample=copy.deepcopy(image)
            masked_sample=self.mask_images_glass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save('./out_pic_case/glassimg%d.jpg'%index)
            # return masked_sample_t, masked_sample_clip, sample, image_label, 2
        if (1):
            sample=copy.deepcopy(image)
            masked_sample=self.mask_images_sunglass(sample)
            # masked_sample=self.crop_images(masked_sample,resize_prob)
            # sample=self.crop_images(sample,resize_prob)
            img = Image.fromarray(masked_sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save('./out_pic_case/sunglass_img%d.jpg'%index)
            # plt.savefig('./out_pic/img%d.jpg'%index)
            # return masked_sample_t, masked_sample_clip, sample, image_label, 3
        if(1):
            sample=copy.deepcopy(image)
            img = Image.fromarray(sample[:,:,[2,1,0]])
            # 保存PIL Image对象为图片
            img.save('./out_pic_case/img%d.jpg'%index)
            # sample=self.crop_images(sample,resize_prob)
            if self.transform is not None:
                sample_t = self.transform(sample)
            if self.preprocess is not None:
             sample_clip=self.preprocess(Image.fromarray(np.uint8(sample)))
            # print('no mask')
            return sample_t,sample_clip,sample_t,image_label, 0
    def crop_images(self,img,prob):
        if(prob>0.5):
            return img
        else:
            # print('size',img.shape)
            new_size=int((img.shape[0]*(1+prob))//2*2)
            # print(new_size)
            new_img=img
            new_img=cv2.resize(img, (new_size, new_size),interpolation=cv2.INTER_LINEAR)
            begin=(new_size-img.shape[0])//2
            end=begin+img.shape[0]
            new_img=new_img[begin:end,begin:end,:]
            return new_img
    def mask_images(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        rs = np.random.randint(-40,40)
        rx = np.random.randint(-10,10)
        # rs=0
        # rx=0
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")

        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.mask_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_glass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.glass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def mask_images_sunglass(self,img):
        
        # get landmarks
        nose = (landmarks[2][0], landmarks[2][1])
        mouth_left = (landmarks[4][0], landmarks[4][1])
        mouth_right = (landmarks[3][0], landmarks[3][1])
        eye_left = (landmarks[1][0], landmarks[1][1])
        eye_right = (landmarks[0][0], landmarks[0][1])
    
        #apply random shift of fakemask
        # rs = np.random.randint(-40,40)
        # rx = np.random.randint(-10,10)
        rs = 0
        rx = 0
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([678+rx,464+rs]), 
                            np.array([548+rx,614+rs]), 
                            np.array([991+rx,664+rs]), 
                            np.array([1009+rx,64+rs]), 
                            np.array([557+rx,64+rs])], dtype="float32")
        '''
        #keypoints of mask image
        src_pts = np.array([np.array([813.5+rx,450+rs+20*1]), 
                            np.array([580+rx,614+rs+20*1]), 
                            np.array([1047+rx,614+rs+20*1]), 
                            np.array([967+rx,150+rs-20*5]), 
                            np.array([660+rx,150+rs-20*5])], dtype="float32")
        #landmark of image
        dst_pts= np.array([np.array([int(nose[0]), int(nose[1])]), 
                           np.array([int(mouth_left[0]), int(mouth_left[1])]), 
                           np.array([int(mouth_right[0]), int(mouth_right[1])]), 
                           np.array([int(eye_right[0]), int(eye_right[1])]), 
                           np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')

        # compute perspective transformation matrix. src_pts -> dst_pts
        # The output matrix is used in next step for the transformation of 
        # the mask to an output-mask which fits to the landmark of the image
        M, _ = cv2.findHomography(src_pts, dst_pts)
    
        # transform the mask to a mask which fits to the image
        transformed_mask = cv2.warpPerspective(
                         self.sunglass_img,
                         M,
                         (img.shape[1], img.shape[0]),
                         None,
                         cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)
 
        # overlay the image with the fitting mask
        alpha_mask = transformed_mask[:, :, 3] / 255
        alpha_image = np.abs(1 - alpha_mask)
        
        # fix mask values
        transformed_mask = transformed_mask / 255 * 100
        
        # add color to masks
        transformed_mask = self.cymk_to_rgb(transformed_mask)
        random_value = np.random.randint(0,150,3)
        transformed_mask = transformed_mask + random_value
        
        for c in range(0, 3):
            img[:, :, c] = (alpha_mask * transformed_mask[:, :, c] + alpha_image * img[:, :, c])

        return img

    def cymk_to_rgb(self, img):
        cyan = img[:,:,0] 
        magenta = img[:,:,1] 
        yellow = img[:,:,2] 
        black = img[:,:,3]
        
        scale = 100
        red = 255*(1.0-(cyan+black)/float(scale))
        green = 255*(1.0-(magenta+black)/float(scale))
        blue = 255*(1.0-(yellow+black)/float(scale))
            
        rgbimg = np.stack((red, green, blue))
        rgbimg = np.moveaxis(rgbimg, 0, 2)
        return rgbimg