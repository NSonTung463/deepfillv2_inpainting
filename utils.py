import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

from  model import network
import torchvision.models as models
import yaml

import matplotlib.pyplot as plt
import math

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    if opt.load_name:
        generator = load_dict(generator, opt.load_name)
    else:
        # Init the networks
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Pre-trained VGG-16
    try:
        vgg16 = torch.load('./vgg16_pretrained.pth')
    except:
        vgg16 = models.vgg16(pretrained=True)
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.named_children() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def save_states(fname, gen, dis, g_optimizer, d_optimizer, n_iter,path_save):
    state_dicts = {'G': gen.state_dict(),
                    'D': dis.state_dict(),
                    'G_optim': g_optimizer.state_dict(),
                    'D_optim': d_optimizer.state_dict(),
                    'n_iter': n_iter}
    torch.save(state_dicts, f"{path_save}/weights/{fname}")
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim



# WARMUP_METHOD ='exp'
LR_MAX=0.05
lr=0.05
N_EPOCHS= 100
N_WARMUP_EPOCHS = 10

def draw_plot(N_EPOCHS,learning_rates,BATCH_SIZE,iters,train_acc_list,losses,path_img):
    plt.title("Training Curve (batch_size={}, lr={})".format(BATCH_SIZE, lr))
    plt.plot(list(range(1,N_EPOCHS+1))  , learning_rates, label='Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Iterations')
    plt.legend()
    plt.savefig(path_img + 'Learning_Rate.png')
    plt.show()
    
    plt.title("Training Curve (batch_size={}, lr={})".format(BATCH_SIZE, lr))
    plt.plot(iters, train_acc_list, label="Train")
    # plt.plot(iters, val_acc_list, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig(path_img + 'Training_Accuracy.png')
    plt.show()

    plt.title("Losese Curve (batch_size={}, lr={})".format(BATCH_SIZE, lr))
    plt.plot(iters, losses, label="Train",color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(path_img + 'Loss.png')
    plt.show()
    
# mnist_train_test = mnist_train[:100]
def lrfn(current_step, num_warmup_steps, lr_max,num_training_steps, num_cycles=0.50,WARMUP_METHOD='exp' ):
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
def plot_lr_schedule(lr_schedule, epochs):
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1

    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])

    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)

    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)

    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()
def get_accuracy(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in torch.utils.data.DataLoader(data, batch_size=len(data)):
            inputs, labels = inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs) # We don't need to run F.softmax
            # loss = criterion(inputs, labels)
            # val_loss += loss.item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        model.train()
        # average_loss = val_loss / len(dataloader)
    return correct / total

def visualize_tensor(tensor, title="Image"):
    # Check if the tensor is a batch tensor and remove the batch dimension if necessary
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    np_img = tensor.numpy()

    # If the tensor is a single-channel (mask), remove the channel dimension for visualization
    if np_img.shape[0] == 1:
        np_img = np_img.squeeze(0)
        plt.imshow(np_img, cmap='gray')
    else:
        # Transpose the dimensions to (H, W, C) for visualization
        np_img = np.transpose(np_img, (1, 2, 0))
        plt.imshow(np_img)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def create_folder_train(name_folder=None):
    runs_path = 'runs'
    os.makedirs(runs_path, exist_ok=True)
    if name_folder is None:
        i = 0
        while True:
            i += 1
            name_train = f'{runs_path}/train{i}'
            if not os.path.exists(name_train):
                os.makedirs(name_train)
                break
    else:
        name_train = f'{runs_path}/{name_folder}'
        os.makedirs(name_train)
    print("All files save in: ", name_train)
    os.makedirs(f'{name_train}/weights')
    os.makedirs(f'{name_train}/sample_images')
    return name_train
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()
def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config
if __name__ == '__main__':
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_training_steps=N_EPOCHS, num_cycles=0.50,) for step in range(N_EPOCHS)]
    plot_lr_schedule(LR_SCHEDULE, N_EPOCHS)
