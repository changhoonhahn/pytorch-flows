'''

script to validate the conditional normalizing flow 


'''
import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision

import datasets
import flows as fnn
import utils


#dat_dir = "/Users/chahah/data/arcoiris/"
dat_dir = "/tigress/chhahn/arcoiris/"
#############################################################################
batch_size      = 100
test_batch_size = 1000
epochs          = 100
learning_rate   = 1e-4
flow            = 'maf'
num_blocks      = 5
seed            = 1

num_hidden      = 1024
act             = 'relu'
#############################################################################

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 4, 'pin_memory': True}
else:
    kwargs = {}

central_pixel = np.zeros((28,28)).astype(bool)
central_pixel[13:17,13:17] = True
central_pixel = central_pixel.flatten()

def load_p_XandY_model(): 
    num_inputs = 28*28#np.sum(central_pixel)
    num_cond_inputs = None

    modules = []

    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    model = fnn.FlowSequential(*modules)
    model.to(device)
    return model 

def load_p_XgivenY_model(): 
    num_inputs = np.sum(~central_pixel)
    num_cond_inputs = np.sum(central_pixel) 

    modules = []
    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    model = fnn.FlowSequential(*modules)
    model.to(device)
    return model 


def load_p_Y_model(): 
    num_inputs = np.sum(central_pixel)
    num_cond_inputs = None

    modules = []
    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    model = fnn.FlowSequential(*modules)
    model.to(device)
    return model 

p_XandY_model   = load_p_XandY_model()
p_XgivenY_model = load_p_XgivenY_model()
p_Y_model       = load_p_Y_model() 

# load model states
state = torch.load(os.path.join(dat_dir, 'p_XgivenY_model_state.pt'), map_location=device)
p_XgivenY_model.load_state_dict(state)

state = torch.load(os.path.join(dat_dir, 'p_XandY_model_state.pt'), map_location=device)
p_XandY_model.load_state_dict(state)

state = torch.load(os.path.join(dat_dir, 'p_Y_model_state.pt'), map_location=device)
p_Y_model.load_state_dict(state)

'''
    fixed_noise = torch.Tensor(batch_size, 28 * 28).normal_()
    # make some validation plots 
    p_XandY_model.eval()
    with torch.no_grad():
        imgs = p_XandY_model.sample(batch_size, noise=fixed_noise).detach().cpu()
        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))

    torchvision.utils.save_image(imgs, os.path.join(dat_dir, 'img_p_XandY_validate.png'), nrow=10)

    blank = torch.zeros(16) - 10 
    p_XgivenY_model.eval() 
    with torch.no_grad():
        imgs = p_XgivenY_model.sample(batch_size, noise=fixed_noise, cond_inputs=blank).detach().cpu()
        full = torch.from_numpy(np.zeros((batch_size, 28*28))).float()
        full[:,~central_pixel] = imgs
        full[:,central_pixel] = blank

        full = torch.sigmoid(full.view(batch_size, 1, 28, 28))

    torchvision.utils.save_image(full, os.path.join(dat_dir, 'img_p_XgivenY_validate.png'), nrow=10)
'''
    
n_val = 100000
noise_y = torch.Tensor(n_val, np.sum(central_pixel)).normal_()
noise_x = torch.Tensor(n_val, np.sum(~central_pixel)).normal_()

p_Y_model.eval() 
p_XgivenY_model.eval() 
with torch.no_grad():
    # sample Y' ~ p_NF(Y) 
    Yp = p_Y_model.sample(n_val, noise=noise_y).detach().cpu()

    # sample X' ~ p_NF(X|Y')
    Xp = p_XgivenY_model.sample(n_val, noise=noise_x, cond_inputs=Yp).detach().cpu()

    XpYp = torch.from_numpy(np.zeros((n_val, 28*28))).float()
    XpYp[:,~central_pixel] = Xp
    XpYp[:,central_pixel] = Yp
    full = torch.sigmoid(XpYp.view(n_val, 1, 28, 28))
torchvision.utils.save_image(full[:100,:,:,:], os.path.join(dat_dir, 'img_p_XpandYp_validate.png'), nrow=10)

p_XandY_model.eval() 
with torch.no_grad(): 
    XpYp = XpYp.to(device)
    # transform (X', Y') with the (X,Y) normalizing flow 
    Zp = p_XandY_model.forward(XpYp, None, mode='direct')[0].detach().cpu() 
    print(Zp)
    full = torch.sigmoid(Zp.view(n_val, 1, 28, 28))
torchvision.utils.save_image(full[:100,:,:,:], os.path.join(dat_dir, 'img_p_Zp_validate.png'), nrow=10)

# save to file 
np.save(os.path.join(dat_dir, 'Zp.npy'), np.array(Zp))

import corner as DFM 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 

fig = DFM.corner(np.array(Zp)[:,:10])
fig.savefig(os.path.join(dat_dir, 'Zp.corner.png'), bbox_inches='tight') 
