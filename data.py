import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch.nn.functional import interpolate


class LoadData(Dataset):

    def __init__(self,path,s=4,fis=144):
        self.data = np.load(path)
        self.data = torch.from_numpy(self.data)
        self.data /= (2**16 - 1)
        shape = self.data.shape
        self.HR = torch.zeros((shape[0]*9,31,144,144))
        
        count = 0
        for i in range(shape[0]):
            for x in range((s+6), shape[2]-(s+6)-fis, fis):
                for y in range((s+6), shape[2]-(s+6)-fis, fis):

                    self.HR[count] = self.data[i,:,x:x+fis,y:y+fis]
                    count += 1

        self.LR = self.down_sample(self.HR)

    
    def down_sample(self, data, s=4):


        data = interpolate(
            data,
            scale_factor=1/s,
            mode='bicubic',
            align_corners=True
        )

        return data


    def __len__(self):
        return self.HR.shape[0]

    
    def __getitem__(self,index):
        return self.LR[index], self.HR[index]

