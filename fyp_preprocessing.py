#145*145*200 to 145*145*11 then sub block lets say 32*32*11 then low and hr pairs
# of 16*16*11 and 32*32*11, 
#do GAN to produce 32*32*11 then try classification.
# Then concatenate and convert to 145*145*200 again for visualization'
from typing import final
from scipy.io import loadmat
from spectral import *
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import cv2 
from PIL import Image 
import os 
import pandas as pd
import numpy as np

data2 = loadmat(r"D:\College\4th year\8th sem\fyp\Indian_pines_corrected.mat")
x2 = data2["indian_pines_corrected"]
data3 = loadmat(r"D:\College\4th year\8th sem\fyp\Indian_pines_gt.mat")
x3 = data3["indian_pines_gt"]
pixel = x2[50,100]
groundt = x3
hisimg = x2
bands = x2[:,:,5]
bands2 = x2[:,:,199]


def extract_pixels(hisimg, groundt):
  q = hisimg.reshape(-1, hisimg.shape[2])
  df = pd.DataFrame(data = q)
  df = pd.concat([df, pd.DataFrame(data = groundt.ravel())], axis=1)
  df.columns= [f'band{i}' for i in range(1, 1+hisimg.shape[2])]+['class']
  return df
  
df = extract_pixels(hisimg, groundt)

from sklearn.decomposition import PCA

pca = PCA(n_components = 75)

principalComponents = pca.fit_transform(df.iloc[:, :-1].values)

ev=pca.explained_variance_ratio_

plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(ev))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


plt.show()

pca = PCA(n_components = 11)
dt = pca.fit_transform(df.iloc[:, :-1].values)
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = groundt.ravel())], axis = 1)
q.columns = [f'PC-{i}' for i in range(1,12)]+['class']

fig = plt.figure(figsize = (30, 15))

for i in range(1, 1+5):
    fig.add_subplot(2,4, i)
    plt.imshow(q.loc[:, f'PC-{i}'].values.reshape(145, 145), cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Band - {i}')
    



q.to_csv('indianpines_11bands.csv', index=False)

elevenbandimg = q

elevenbandimg.drop('class', inplace=True, axis=1)
final11 = elevenbandimg.loc[:,:].values.reshape(145,145,11)
# print(final11.shape)

# randomly generate 32x32x11 images 

patches = patchify(final11, (32, 32, 11), step = 1)
print(patches.shape)
resized_patches = np.zeros((114, 114, 1, 16, 16, 11))
# copy_array = np.array(patches).copy()
# print(copy_array.shape)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        for k in range(patches.shape[2]):
            image = patches[i][j][k] 
            image.shape
            resized = cv2.resize(image, (16, 16))
            resized_patches[i][j][k] = resized 

os.chdir(r'D:\College\4th year\8th sem\fyp\step 1 patches\hr')
for x in range(patches.shape[0]):
    for y in range(patches.shape[1]):
        for z in range(patches.shape[2]):
            for b in range(11):
                image1 = patches[x][y][z][:,:,b] 
                cmap = plt.get_cmap('jet')
                cv2.imwrite('image_' + str(x) + str(y) + str(b) + ".png",image1)

os.chdir(r'D:\College\4th year\8th sem\fyp\step 1 patches\lr')
for x in range(resized_patches.shape[0]):
    for y in range(resized_patches.shape[1]):
        for z in range(resized_patches.shape[2]):
            for b in range(11):
                image1 = resized_patches[x][y][z][:,:,b] 
                cmap = plt.get_cmap('jet')
                cv2.imwrite('image_' + str(x) + str(y) + str(b) + ".png",image1)

