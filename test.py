from pathlib import Path
from net import Generator,Discriminator
import torch
from torch.utils.data import DataLoader
from G import OUT_DIR,TEST_DATA_PATH
from data import LoadData
from utils import *


BATCH_SIZE = 9
FAKE_HR = torch.zeros([6*9,31,144,144])
HR = torch.zeros([6*9,31,144,144])
PSNR = 0
SAMs = 0

if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test decice is {}'.format(device))

    g_model = Generator(BATCH_SIZE).to(device)
    state_dict_g = g_model.state_dict()

    d_model = Discriminator(BATCH_SIZE).to(device)
    state_dict_d = d_model.state_dict()

    g_weight = OUT_DIR.joinpath('g_model.pth')

    d_weight = OUT_DIR.joinpath('d_model.pth')

    for n, p in torch.load(g_weight, map_location=lambda storage, loc: storage).items():
        if n in state_dict_g.keys():
            state_dict_g[n].copy_(p)
        else:
            raise KeyError(n)

    for n, p in torch.load(d_weight, map_location=lambda storage, loc: storage).items():
        if n in state_dict_d.keys():
            state_dict_d[n].copy_(p)
        else:
            raise KeyError(n)


    g_model.eval()
    d_model.eval()

    test_data = DataLoader(
            LoadData(TEST_DATA_PATH),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers= 2, 
            pin_memory= True,
            drop_last= True,
        )

    count = 0
    for lr,hr in test_data:

        lr = lr.reshape((lr.shape[0],1,lr.shape[1],lr.shape[2],lr.shape[3]))
        lr = lr.to(device)
        hr = hr.reshape((hr.shape[0],1,hr.shape[1],hr.shape[2],hr.shape[3]))
        hr = hr.to(device)


        with torch.no_grad():

            fake_hr = g_model(lr)
            fake_hr = torch.squeeze(fake_hr)
            hr = torch.squeeze(hr)

            hr = hr.cpu()
            fake_hr = fake_hr.cpu()

            psnr = PSNR_GPU(hr,fake_hr)
            sam = SAM(hr,fake_hr)
            print('img : {} psnr : {:.4f}  sam : {:.4f}'.format(
                count+1,psnr,sam
                ))
            PSNR += psnr
            SAMs += sam

        FAKE_HR[count*9:(count+1)*9] = fake_hr
        HR[count*9:(count+1)*9] = hr
        
        count += 1
    print(PSNR / 6, SAMs / 6)
    torch.save(FAKE_HR, OUT_DIR.joinpath('test_fake_hr.pth'))
    torch.save(HR, OUT_DIR.joinpath('hr.pth'))
            


            




