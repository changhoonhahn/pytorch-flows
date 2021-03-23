import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def save_moons_plot(epoch, best_model, dataset):
    # generate some examples
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(500).detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:, 0], dataset.val.x[:, 1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Synth data')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    plt.savefig('plots/plot_{:03d}.png'.format(epoch))
    plt.close()


batch_size = 100
fixed_noise = torch.Tensor(batch_size, 28 * 28).normal_()
#y = torch.arange(batch_size).unsqueeze(-1) % 10
#y_onehot = torch.FloatTensor(batch_size, 10)
#y_onehot.zero_()
#y_onehot.scatter_(1, y, 1)


def save_images(epoch, best_model, name):
    best_model.eval()
    with torch.no_grad():
        imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()

        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))
    
    torchvision.utils.save_image(imgs, '/tigress/chhahn/arcoiris/img_%s_%i.png' % (name, epoch), nrow=10)


def save_images_p_XandY(epoch, best_model):
    best_model.eval()
    with torch.no_grad():
        imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()
        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))
    
    torchvision.utils.save_image(imgs, '/tigress/chhahn/arcoiris/img_p_XandY_%i.png' % epoch, nrow=10)


def save_images_p_Y(epoch, best_model):
    best_model.eval()
    with torch.no_grad():
        imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()
        imgs = torch.sigmoid(imgs.view(batch_size, 1, 4, 4))
    
    torchvision.utils.save_image(imgs, '/tigress/chhahn/arcoiris/img_p_Y_%i.png' % epoch, nrow=10)


central_pixel = np.zeros((28,28)).astype(bool)
central_pixel[13:17,13:17] = True 
central_pixel = central_pixel.flatten() 
blank = torch.zeros(16)
def save_images_p_XgivenY(epoch, best_model):
    best_model.eval()
    with torch.no_grad():
        imgs = best_model.sample(batch_size, noise=fixed_noise, cond_inputs=blank).detach().cpu()
        full = torch.from_numpy(np.zeros((batch_size, 28*28))).float()
        full[:,~central_pixel] = imgs

        full = torch.sigmoid(full.view(batch_size, 1, 28, 28))
    
    torchvision.utils.save_image(full, '/tigress/chhahn/arcoiris/img_p_XgivenY_%i.png' % epoch, nrow=10)
