import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance


def random_blur(img):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def random_rotation(img):
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def random_noise(img):
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def random_contrast(img):
    factor = random.uniform(0.6, 1.4)
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_img)
    return np.array(enhancer.enhance(factor))


def random_occlusion(img):
    h, w = img.shape[:2]
    x1 = random.randint(0, w // 2)
    y1 = random.randint(0, h // 2)
    x2 = x1 + random.randint(30, 120)
    y2 = y1 + random.randint(20, 80)
    img[y1:y2, x1:x2] = 255
    return img


def apply_random_augmentations(img):
    funcs = [random_blur, random_rotation, random_noise, random_contrast, random_occlusion]
    num = random.randint(1, 3)
    for f in random.sample(funcs, num):
        img = f(img)
    return img
