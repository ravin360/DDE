import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def load_kitti_data(dataset_root):
    data = {"left_images": [], "right_images": [], "disparity": [], "calib": {}}
    left_dir = os.path.join(dataset_root, 'image_2')
    right_dir = os.path.join(dataset_root, 'image_3')
    disp_dir = os.path.join(dataset_root, 'disp_occ_0')


    for filename in sorted(os.listdir(left_dir)):
        if filename.endswith("10.png"):
            left_image = cv2.imread(os.path.join(left_dir, filename))
            disp_path = os.path.join(disp_dir, filename)
            disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
            data["left_images"].append(left_image)
            data["disparity"].append(disparity)
        elif filename.endswith("11.png"):
            right_image = cv2.imread(os.path.join(right_dir, filename))
            data["right_images"].append(right_image)
    return data

def preprocess_image(image, size=(512,256)):
    image = cv2.resize(image, size)
    return image

def preprocess_label(label, size=(512, 256)):
    label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    label = label / np.max(label) if np.max(label) > 0 else label
    return label

def load_dataset(leftimg_dir, label_dir, rightimg_dir, size=(512, 256)):
    leftimgs, labels, rightimgs = [], [], []
    for img in leftimg_dir:
        leftimg = preprocess_image(img, size)
        leftimgs.append(leftimg)
    for img in rightimg_dir:
        rightimg = preprocess_image(img, size)
        rightimgs.append(rightimg)
    for img in label_dir:
        label = preprocess_label(img, size)
        labels.append(label)
    return np.array(leftimgs), np.array(labels), np.array(rightimgs)


def binarize_image(img):
    rows, cols, _ = img.shape
    binary_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for channel in range(3):
                binary_value = 0
                for k, (dx, dy) in enumerate(neighbors):
                    if img[i, j, channel] > img[i + dx, j + dy, channel]:
                        binary_value |= (1 << (7 - k))
                binary_image[i, j, channel] = binary_value
    return Image.fromarray(binary_image)


def rec_fld(image, patch_size):
    binary_rec=[]
    height, width, rgb = image.shape
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            binary_rec.append(binarize_image(patch))
    return np.array(binary_rec)


def calculate_max_shift(left_img, right_img, max_disparity=192):
    m_arr = []
    for d in range(1, max_disparity):
        shifted_right_img = np.roll(right_img, shift=-d, axis=1)
        m_arr.append(np.mean(np.abs(left_img[:, d:] - shifted_right_img[:, d:])))
    m_arr.sort()
    best_disparity = m_arr[0]
    return best_disparity


def train_rec(l_img, r_img):
    max_shift = calculate_max_shift(l_img, r_img)
    patches = [55, 25, 15]
    p_values = []
    for p in patches:
        l_patch = rec_fld(l_img, patch_size=p)
        r_patch = rec_fld(r_img, patch_size=p)
        rectified_r_patch = np.zeros_like(r_patch)
        for i in range(r_patch.shape[0]):
            shift = min(max_shift, r_patch.shape[2])
            rectified_r_patch[i] = np.roll(r_patch[i], shift=-shift, axis=1)

        xor = np.bitwise_xor(l_patch, rectified_r_patch)
        p_values.append(np.sum(xor))
    return tuple(p_values)

dataset_root = "C:/Me/Project fy/Datasets/KITTI2015a/training"
kitti_data = load_kitti_data(dataset_root)
print(f"Loaded {len(kitti_data['left_images'])} image pairs.")
print(f"Loaded {len(kitti_data['disparity'])} disparity maps.")
left_img, disparity, right_img = load_dataset(kitti_data['left_images'],kitti_data['disparity'],kitti_data['right_images'], size=(512, 256))
print(f"Preprocessed {len(left_img)} images and {len(disparity)} labels.")
print(left_img[0].shape)

plt.figure(figsize=(10, 5))
plt.imshow(left_img[0])
plt.title("Preprocessed Image")
plt.show()
plt.imshow(disparity[0], cmap='gray')
plt.title("Preprocessed Depth Map")
plt.show()

train_left_tensor = torch.tensor(left_img,dtype=torch.float32).permute(0, 3, 1, 2)
train_right_tensor = torch.tensor(right_img,dtype=torch.float32).permute(0, 3, 1, 2)
train_disp_tensor = torch.tensor(disparity,dtype=torch.float32).unsqueeze(1)
print(f"Train images tensor shape: {train_left_tensor.shape}")
print(f"Train labels tensor shape: {train_disp_tensor.shape}")

mc1, mc2, mc3, dispk, m1 = [], [], [], [], []
for l, r in zip(left_img, right_img):
    a,b,c= train_rec(l,r)
    d = calculate_max_shift(l, r)
    mc1.append(a)
    mc2.append(b)
    mc3.append(c)
    dispk.append(d)
    print(a)

m = []
for i in range(len(dispk)):
    channel1 = np.full((256,512), dispk[i] / max(dispk), dtype=np.float32)
    channel2 = np.full((256,512), (1 - (mc1[i] / max(mc1))) / 3 + (1 - (mc2[i] / max(mc2))) / 3 + (1 - (mc3[i] / max(mc3))) / 3, dtype=np.float32)
    m1 = np.stack([channel1, channel2], axis=0)
    m.append(m1)
m = np.array(m, dtype=np.float32)

print("Shape of m:", m.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
