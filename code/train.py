import argparse
import os,time
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Dataset_generator
from tqdm import tqdm
from loss import GeneratorLoss
from model import Generator,Discriminator
from utils import ssim,display_transform
from utils import read_h5,save_h5
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train Super Resolution AND Denoising Model")
    parser.add_argument("--upscale_factor",default = 1,type = int,help = "super resolution upscale factor")
    parser.add_argument("--num_epochs",default = 150,type = int,help = "train epoch number")
    parser.add_argument("--data_train", default = "../data/train/1", type=str, help="high resolution data path of train set")
    parser.add_argument("--data_val", default = "../data/test/1", type=str, help="high resolution data path of val set")
    parser.add_argument("--lr_D", default = 1e-4, type = float,help="learning rate of discriminator")
    parser.add_argument("--lr_G", default = 1e-4, type = float,help="learning rate of generator")
    parser.add_argument("--batch_size", default = 4, type = int,help="batch size of train dataset")
    parser.add_argument("--out_path", default="../result/MSRD-GAN", type=str, help="the path of save file")
    parser.add_argument("--gpu_id",default = 0,type = int,help = "GPU id of used,eg.0,1,2")
    parser.add_argument("--adversarial_index", default = 0.01, type = float, help="adversarial loss weights in generator loss")
    parser.add_argument("--perception_index", default = 0.06, type = float, help="perception loss weights in generator loss")
    opt  = parser.parse_args()
    # main(opt)

 # define hyperparameters
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    DATA_TRAIN = opt.data_train
    DATA_VAL = opt.data_val
    BATCH_SIZE = opt.batch_size
    OUT_PATH = opt.out_path
    LR_D = opt.lr_D
    LR_G = opt.lr_G
    GPU_ID = opt.gpu_id
    ADV_INDEX= opt.adversarial_index
    PECP_INDEX = opt.perception_index
    torch.cuda.set_device(GPU_ID)
    
   
    
    # load training data
    train_set = Dataset_generator(data_path = DATA_TRAIN,sigma=25)
    val_set = Dataset_generator(data_path = DATA_VAL,sigma=25,train = False)
    train_loader = DataLoader(dataset = train_set,batch_size = BATCH_SIZE,shuffle = True)
    val_loader = DataLoader(dataset = val_set,batch_size = 1,shuffle = False)

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # save hyperparameter
    argsDict = opt.__dict__
    with open(OUT_PATH + "/hyperparameter.txt","w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg,value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)

    model_path = os.path.join(OUT_PATH, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    statistics_path = os.path.join(OUT_PATH, "statistics")
    if not os.path.exists(statistics_path):
        os.makedirs(statistics_path)

    # define network,loss,optimizer
    netG = Generator(scale_factor = UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # loss of generator
    generator_criterion = GeneratorLoss(adversarial_index = ADV_INDEX,perception_index = PECP_INDEX)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(),lr = LR_G)
    optimizerD = optim.Adam(netD.parameters(),lr = LR_D)

    # train
    results_train = {'psnr': [], 'ssim': []}
    results_val = {'psnr': [], 'ssim': []}
    save_metric = 0.0
    
    start_time = time.time()  

    # epoch
    for epoch in range(1,NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, "mse":0,"psnr":0,"ssim":0}

        netG.train()
        netD.train()
        # train
        for train_data,train_target in train_bar:
            g_update_first = True
            batch_size = train_data.size(0)
            running_results["batch_sizes"] += batch_size

            # (1) update D network
            real_img = Variable(train_target)
            z = Variable(train_data)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph = True)
            optimizerD.step()

            # (2) Uptrain_data G network
            netG.zero_grad()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            g_loss = generator_criterion(fake_out,fake_img,real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            optimizerG.step()

            # mse
            batch_mse = ((fake_img - real_img) ** 2).data.mean()
            running_results["mse"] += batch_mse * batch_size
            # psnr
            running_results["psnr"] += 10 * log10((real_img.max()**2) / batch_mse) * batch_size
            # ssim
            batch_ssim = ssim(fake_img,real_img).item()
            running_results["ssim"] += batch_ssim * batch_size


            train_bar.set_description(desc='[%d/%d] PSNR: %.4f SSIM: %.4f' % (
                  epoch, NUM_EPOCHS, running_results["psnr"] / running_results["batch_sizes"],
                  running_results["ssim"] / running_results["batch_sizes"]
                  ))
        train_elapsed_time = time.time() - start_time 
        print('\n'+'train_elapsed_time = ' + str(train_elapsed_time))


        # val
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_g_loss = 0.0
            val_d_loss = 0.0
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_data,val_target in val_bar:
                batch_size = val_data.size()[0]
                valing_results["batch_sizes"] += batch_size
                if torch.cuda.is_available():
                    lr = val_data.cuda()
                    hr = val_target.cuda()
                sr = netG(lr)
                real_out = netD(hr).mean()
                fake_out = netD(sr).mean()
                loss_d = 1 - real_out + fake_out
                loss_g = generator_criterion(fake_out, sr, hr)

                # mse
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                # ssim and psnr
                batch_ssim = ssim(sr,hr).item()
                valing_results["ssim"] += batch_ssim * batch_size
                psnr = 10 * log10((hr.max() ** 2) / batch_mse)
                valing_results["psnr"] += psnr * batch_size
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results["psnr"] / valing_results["batch_sizes"], valing_results["ssim"] / valing_results["batch_sizes"]))

        val_elapsed_time = time.time() - start_time - train_elapsed_time
        print('\n'+'val_elapsed_time = ' + str(val_elapsed_time))

        # save model parameters
        if epoch % 5 == 0 and epoch != 0:
            torch.save(netG.state_dict(),model_path + "/netG_epoch_%d.pth" % (epoch))
            torch.save(netD.state_dict(),model_path + '/netD_epoch_%d.pth' % (epoch))
            
        metric = (valing_results["psnr"] / valing_results["batch_sizes"]) * (valing_results["ssim"] / valing_results["batch_sizes"])
        if metric > save_metric:
            torch.save(netG.state_dict(), model_path + "/netG_bestmodel.pth")
            torch.save(netD.state_dict(), model_path + '/netD_bestmodel.pth')
            save_metric = metric
                
        # save psnr\ssim
        # train
        results_train['psnr'].append(running_results['psnr'] / running_results['batch_sizes'])
        results_train['ssim'].append(running_results['ssim'] / running_results['batch_sizes'])

        # val
        results_val["psnr"].append(valing_results["psnr"] / valing_results["batch_sizes"])
        results_val["ssim"].append(valing_results["ssim"] / valing_results["batch_sizes"])

        data_frame_train = pd.DataFrame(
            data = {'PSNR': results_train['psnr'], 'SSIM': results_train['ssim']},
            index = range(1,epoch + 1)
        )
        data_frame_val = pd.DataFrame(
            data={'PSNR': results_val['psnr'], 'SSIM': results_val['ssim']},
            index=range(1, epoch + 1)
        )
        data_frame_train.to_csv(statistics_path + "/train_results.csv",index_label="Epoch")
        data_frame_val.to_csv(statistics_path + "/val_results.csv", index_label="Epoch")









