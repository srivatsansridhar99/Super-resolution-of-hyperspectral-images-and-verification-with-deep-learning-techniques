import torch
import torch.nn as nn
import torch.optim as optim
from net import Generator, Discriminator,Spe_loss,TVLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
from G import *
from icvl_data import LoadData
from utils import SAM, PSNR_GPU, get_paths
from pathlib import Path


EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    g_model = Generator(BATCH_SIZE).to(device)
    d_model = Discriminator(BATCH_SIZE).to(device)

    d_criterion = nn.BCELoss()
    criterion = {
        'l1' : nn.L1Loss(),
        'ltv' : TVLoss(),
        'ls' : Spe_loss(),
        'la' : nn.BCELoss(),
    }


    g_optimizer = optim.Adam(
        g_model.parameters(),
        lr = LR
    )
    d_optimizer = optim.SGD(
        d_model.parameters(),
        lr = LR
    )

    sorce = {
        'd_loss':0.0,
        'g_loss':0.0,
        'real_sorce':0.0,
        'fake_sorce':0.0
    }


    best_sorce = {
        'psnr'  : 0.0,
        'sam'   : 180.0,
        'epoch' : 0,
    }

    train_paths, val_paths, _ = get_paths()

    for epoch in range(EPOCHS):

        train_data = DataLoader(
            LoadData(train_paths,'train'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers= 2, 
            pin_memory= True,
            drop_last= True,
        )

        val_data = DataLoader(
            LoadData(val_paths,'val'),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers= 2, 
            pin_memory= True,
            drop_last= True,
        )

        count = 0
        for lr, hr in train_data:
            lr = lr.reshape((lr.shape[0],1,lr.shape[1],lr.shape[2],lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0],1,hr.shape[1],hr.shape[2],hr.shape[3]))
            hr = hr.to(device)

            real_labels = torch.ones(BATCH_SIZE).to(device)
            fake_labels = torch.zeros(BATCH_SIZE).to(device)

            output = d_model(hr)
            d_loss_real = d_criterion(torch.squeeze(output),real_labels)
            real_sorce = output
            sorce['real_sorce'] = real_sorce.mean().item()

            fake_hr = g_model(lr)
            output = d_model(fake_hr)
            d_loss_fake = d_criterion(torch.squeeze(output),fake_labels)
            fake_sorce = output
            sorce['fake_sorce'] = fake_sorce.mean().item()

            d_loss = (d_loss_real + d_loss_fake) / 2
            sorce['d_loss'] = d_loss.item()
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_hr = g_model(lr)
            output = d_model(fake_hr)

            fake_hr = torch.squeeze(fake_hr)
            hr = torch.squeeze(hr)
            g_loss = criterion['l1'](fake_hr,hr) + \
                1e-3 * d_criterion(torch.squeeze(output),real_labels)
            sorce['g_loss'] = g_loss


            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print('EPOCH : {} step : {} \
d_loss : {:.4f} g_loss : {:.4f} \
real_sorce {:.4f} fake_sorce {:.4f}'.format(
                    epoch,count+1,
                    sorce['d_loss'],sorce['g_loss'], 
                    sorce['real_sorce'],sorce['fake_sorce']
                ))
            count += 1



        g_model.eval()
        d_model.eval()
        val_count = 0
        val_psnr = 0
        val_sam = 0
        for lr,hr in val_data:

            lr = lr.reshape((lr.shape[0],1,lr.shape[1],lr.shape[2],lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0],1,hr.shape[1],hr.shape[2],hr.shape[3]))
            hr = hr.to(device)

            with torch.no_grad():

                fake_hr = g_model(lr)
                fake_hr = torch.squeeze(fake_hr)
                hr = torch.squeeze(hr)

                fake_hr = fake_hr.cpu()
                hr = hr.cpu()

                psnr = PSNR_GPU(hr,fake_hr)
                val_psnr += psnr
                sam = SAM(hr,fake_hr)
                val_sam += sam

                print('val epoch : {} step : {} psnr : {:.4f}  sam : {:.4f}'.format(
                    epoch,val_count+1,psnr,sam
                ))

                val_count += 1

        print('val averagr psnr : {:.4f} sam : {:.4f}'.format(
            val_psnr/(val_count),
            val_sam/(val_count))
            )

        if val_psnr/(val_count+1) > best_sorce['psnr']:
            best_sorce['psnr'] = val_psnr/(val_count)

            torch.save(copy.deepcopy(g_model.state_dict()),OUT_DIR.joinpath('icvl_g_model.pth'))
            torch.save(copy.deepcopy(d_model.state_dict()),OUT_DIR.joinpath('icvl_d_model.pth'))


    