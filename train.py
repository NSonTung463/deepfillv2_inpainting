import os
import time
import argparse
import torch
import torchvision as tv
import torchvision.transforms as T
import yaml
import model.losses as gan_losses
import utils
import torch.nn as nn
from data.dataset import ImageDataset
from model.networks import Generator, Discriminator
from tqdm import tqdm
import misc
import utils
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="configs/train.yaml", help="Path to yaml config file")
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0, 1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name', type = str, default = '', help = 'load model name')
    # Training parameters
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 4e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 100, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 10, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of valid loss of AdaReconL1Loss; 0 is recommended')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "./dataset//images", help = 'the training folder')
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    # Load the YAML configuration file
    with open('configs/train.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    # Add each configuration item to the parser
    for key, value in cfg.items():
        # Determine the type of argument to add based on the type of value in the YAML
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=lambda x: (str(x).lower() == 'true'), default=value)
        elif isinstance(value, list):
            parser.add_argument(f'--{key}', type=str, default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    config = parser.parse_known_args()[0]
    
    
    path_save = utils.create_folder_train()
    # generator = utils.create_generator(config)
    # discriminator = utils.create_discriminator(config)
    # perceptualnet = utils.create_perceptualnet()
    cnum_in = config.img_shapes[2]
    
    generator = Generator(cnum_in=cnum_in+2, cnum_out=cnum_in, cnum=48, return_flow=False)
    discriminator = Discriminator(cnum_in=cnum_in+1, cnum=64)
    
    
    device = torch.device('cuda' if torch.cuda.is_available()
                            and config.use_cuda_if_available else 'cpu')
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    generator.train()
    discriminator.train()
    losses = {}
    
    losses_log = {'d_loss':   [],
                'g_loss':   [],
                'ae_loss':  [],
                'ae_loss1': [],
                'ae_loss2': [],
                }
    
    
    train_dataset = ImageDataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=config.num_workers,
                                                pin_memory=True)

    # optimizers
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

    L1Loss = nn.L1Loss()
    best_loss = 99999
    if config.tb_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)
    n_iter = 1
    
# losses
if config.gan_loss == 'hinge':
    gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g
elif config.gan_loss == 'ls':
    gan_loss_d, gan_loss_g = gan_losses.ls_loss_d, gan_losses.ls_loss_g
else:
    raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")
LR_SCHEDULE = [utils.lrfn(step, num_warmup_steps=config.nwarmup, lr_max=config.lr_max,num_training_steps=config.epochs, num_cycles=config.num_cycles) for step in range(config.epochs)]
for epoch in range(config.epochs):
    if config.warmup_status == True:
        for g_param_group,d_param_group in zip(g_optimizer.param_groups,d_optimizer.param_groups):
            g_param_group['lr'] = LR_SCHEDULE[epoch]
            d_param_group['lr'] = LR_SCHEDULE[epoch]
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}', unit='batch')
    for batch_idx,(img,mask) in enumerate(progress_bar):
        
        img = img.to(device)
        mask = mask.to(device)
        
        batch_real  = img
        
        # prepare input for generator
        batch_incomplete = batch_real*(1.-mask)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1].to(device)
        x = torch.cat([batch_incomplete, ones_x, ones_x*mask], axis=1)
        
        # generate inpainted images
        x1, x2 = generator(x, mask)
        batch_predicted = x2
        
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        
        # D training steps:
        batch_real_mask = torch.cat((batch_real, mask), dim=1)
        batch_filled_mask = torch.cat((batch_complete.detach(), mask), dim=1)
        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))

        d_real_gen = discriminator(batch_real_filled)
        d_real, d_gen = torch.split(d_real_gen, config.batch_size)

        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # update D parameters
        d_optimizer.zero_grad()
        losses['d_loss'].backward()
        d_optimizer.step()

        # G training steps:
        losses['ae_loss1'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x1)))
        losses['ae_loss2'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x2)))
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_predicted
        batch_gen = torch.cat((batch_gen, mask), dim=1)

        d_gen = discriminator(batch_gen)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()
        # LOGGING
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())
        n_iter+=1
        
        
        if config.tb_logging \
            and config.save_imgs_to_tb_iter \
            and n_iter % config.save_imgs_to_tb_iter == 0:
            viz_images = [misc.pt_to_image(batch_complete),
                            misc.pt_to_image(x1), misc.pt_to_image(x2)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                        for images in viz_images]
            writer.add_image(
                "Inpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 1", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 2", img_grids[2], global_step=n_iter, dataformats="CHW")
        # save example image grids to disk
        if config.save_imgs_to_disc_iter \
            and n_iter % config.save_imgs_to_disc_iter == 0:
            viz_images = [misc.pt_to_image(batch_real), 
                            misc.pt_to_image(batch_complete)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                                            for images in viz_images]
            tv.utils.save_image(img_grids, 
            f"{path_save}/images/iter_{n_iter}.png", 
            nrow=2)
        if losses['g_loss'].item() < best_loss:
            best_loss = losses['g_loss'].item()
            utils.save_states('best.pth', generator, discriminator, g_optimizer, d_optimizer, n_iter,path_save)
        utils.save_states('last.pth', generator, discriminator, g_optimizer, d_optimizer, n_iter,path_save)
        progress_bar.set_postfix(g_loss=losses['g_loss'].item(), d_loss=losses['d_loss'].item(),ae_loss1=losses['ae_loss1'].item(), ae_loss2=losses['ae_loss2'].item())
    # print(f'loss: {loss.item()}, GAN_Loss={GAN_Loss.item()},loss_D={loss_D.item()}, PerceptualLoss: {second_PerceptualLoss.item()},first maskLoss: {first_MaskL1Loss.item()}, Second_maskLoss: {second_MaskL1Loss.item()}')