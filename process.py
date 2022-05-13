from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from utils import calc_psnr, SAM

bic_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/bic_icvl_img7.png'
hr_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/hr_icvl_img7.png'
gan_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/process_icvl_img7.png'

paths = [gan_path,bic_path]

hr = Image.open(hr_path)
hr = np.array(hr)
hr = hr.astype(np.float32)
hr = torch.from_numpy(hr)

for path in paths:

    img = Image.open(path)
    img = np.array(img)
    img = img.astype(np.float32)
    img = torch.from_numpy(img)  
    print((calc_psnr(hr,img),SAM(hr,img)))