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

#############################################################################
# read in data 
dat_dir = "/tigress/chhahn/arcoiris/provabgs_cnf/"

thetas = np.load(os.path.join(dat_dir, 'train.thetas.npy')) # SPS parameters
xphoto = np.load(os.path.join(dat_dir, 'train.xphoto.npy')) # FM u,g,r,i,z photometry
sigmax = np.load(os.path.join(dat_dir, 'train.sigmax.npy')) # photometric noise 

# posterior will be conditioned on photometry and photometric noise
condit = np.concatenate([xphoto, sigmax], axis=1) 

# setup training and validation set 
Ntrain = int(np.floor(0.8 * thetas.shape[0]))
Nvalid = int(np.floor(0.1 * thetas.shape[0]))
Ntest  = int(np.floor(0.1 * thetas.shape[0]))

train_tensor    = torch.from_numpy(thetas[:Ntrain,:])
train_cond      = torch.from_numpy(condit[:Ntrain,:])
train_dataset   = torch.utils.data.TensorDataset(train_tensor, train_cond)

valid_tensor    = torch.from_numpy(thetas[Ntrain:Ntrain+Nvalid,:])
valid_cond      = torch.from_numpy(condit[Ntrain:Ntrain+Nvalid,:])
valid_dataset   = torch.utils.data.TensorDataset(valid_tensor, valid_cond)

# number of conditional inputs 
num_inputs      = thetas.shape[1]
num_cond_inputs = condit.shape[1]

# set up loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=False, drop_last=False, **kwargs)


#############################################################################
# set up MAF 
modules = []
for _ in range(num_blocks):
    modules += [
        fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
        fnn.BatchNormFlow(num_inputs),
        fnn.Reverse(num_inputs)
    ]
model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

#############################################################################
# train
#############################################################################
# adam optimizer 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

writer = SummaryWriter(comment="provabgs_cnf")
global_step = 0

def train(epoch):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(data.device),
            train_loader.dataset.tensors[1].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1

def validate(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            cond_data = data[1].float()
            cond_data = cond_data.to(device)

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(epochs):
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

    # save training checkpoint
    torch.save({'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()},
        os.path.join(dat_dir, 'provabgs_cnf.checkpoint.pt'))
    # save model only
    torch.save(model.state_dict(), 
            os.path.join(dat_dir, 'provabgs_cnf.model_state.pt'))
