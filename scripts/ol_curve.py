import os

import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == '__main__':
    device = torch.device("cuda:0")
    dir_save = r'../save/online_fm-hp_fm-Spine/RecON'

    MEA, FDR, ADR, MD, SD, HD = [], [], [], [], [], []
    LOSS_PSC, LOSS_D, LOSS_G, LOSS_G_GAS, LOSS_FCC = [], [], [], [], []

    for file in sorted(os.listdir(dir_save)):
        if not file.startswith('value_'):
            continue
        value = torch.load(os.path.join(dir_save, file), map_location=device)
        mea, fdr, adr, md, sd, hd = [], [], [], [], [], []
        for idx, loss in enumerate(value['loss']):
            mea.append(loss['MEA'])
            fdr.append(loss['FDR'])
            adr.append(loss['ADR'])
            md.append(loss['MD'])
            sd.append(loss['SD'])
            hd.append(loss['HD'])
        MEA.append(torch.tensor(mea, device=device))
        FDR.append(torch.tensor(fdr, device=device))
        ADR.append(torch.tensor(adr, device=device))
        MD.append(torch.tensor(md, device=device))
        SD.append(torch.tensor(sd, device=device))
        HD.append(torch.tensor(hd, device=device))

        if 'train_loss' in value and len(value['train_loss']) > 0:
            loss_psc, loss_d, loss_g, loss_g_gas, loss_fcc = [], [], [], [], []
            for tl in value['train_loss']:
                loss_psc.append(tl.get('loss_psc', float('nan')))
                loss_d.append(tl.get('loss_d', float('nan')))
                loss_g.append(tl.get('loss_g', float('nan')))
                loss_g_gas.append(tl.get('loss_g_gas', float('nan')))
                loss_fcc.append(tl.get('loss_fcc', float('nan')))
            LOSS_PSC.append(loss_psc)
            LOSS_D.append(loss_d)
            LOSS_G.append(loss_g)
            LOSS_G_GAS.append(loss_g_gas)
            LOSS_FCC.append(loss_fcc)

    MEA = torch.mean(torch.stack(MEA, dim=0), dim=0).cpu().numpy()
    FDR = torch.mean(torch.stack(FDR, dim=0), dim=0).cpu().numpy()
    ADR = torch.mean(torch.stack(ADR, dim=0), dim=0).cpu().numpy()
    MD = torch.mean(torch.stack(MD, dim=0), dim=0).cpu().numpy()
    SD = torch.mean(torch.stack(SD, dim=0), dim=0).cpu().numpy()
    HD = torch.mean(torch.stack(HD, dim=0), dim=0).cpu().numpy()
    x = np.linspace(0, len(MEA) - 1, len(MEA))

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title('MEA')
    plt.plot(x, MEA)
    plt.subplot(2, 3, 2)
    plt.title('FDR')
    plt.plot(x, FDR)
    plt.subplot(2, 3, 3)
    plt.title('ADR')
    plt.plot(x, ADR)
    plt.subplot(2, 3, 4)
    plt.title('MD')
    plt.plot(x, MD)
    plt.subplot(2, 3, 5)
    plt.title('SD')
    plt.plot(x, SD)
    plt.subplot(2, 3, 6)
    plt.title('HD')
    plt.plot(x, HD)

    if LOSS_PSC:
        loss_psc_mean = np.nanmean(np.array(LOSS_PSC, dtype=float), axis=0)
        loss_d_mean   = np.nanmean(np.array(LOSS_D,   dtype=float), axis=0)
        loss_g_mean   = np.nanmean(np.array(LOSS_G,   dtype=float), axis=0)
        loss_g_gas_mean = np.nanmean(np.array(LOSS_G_GAS, dtype=float), axis=0)
        loss_fcc_mean = np.nanmean(np.array(LOSS_FCC, dtype=float), axis=0)
        xl = np.arange(1, len(loss_psc_mean) + 1)

        plt.figure(figsize=(12, 8))
        plt.suptitle('Training Loss Curves (mean over scans)')

        plt.subplot(2, 3, 1)
        plt.title('PSC Loss')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_psc_mean)

        plt.subplot(2, 3, 2)
        plt.title('Discriminator Loss (GAS_d)')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_d_mean)

        plt.subplot(2, 3, 3)
        plt.title('Generator Loss (GAS_g)')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_g_gas_mean)

        plt.subplot(2, 3, 4)
        plt.title('FCC Loss')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_fcc_mean)

        plt.subplot(2, 3, 5)
        plt.title('Total Generator Loss')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_g_mean)

        plt.subplot(2, 3, 6)
        plt.title('All Losses')
        plt.xlabel('Epoch')
        plt.plot(xl, loss_psc_mean, label='PSC')
        plt.plot(xl, loss_g_gas_mean, label='GAS_g')
        plt.plot(xl, loss_fcc_mean, label='FCC')
        plt.plot(xl, loss_g_mean, label='G total')
        plt.legend()

        plt.tight_layout()

    plt.show()