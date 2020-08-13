import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch



# this class logs the error and predictions, creates visualizations
class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        # SummaryWriter class is your main entry to log data for consumption and visualization
        # by TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    # function that writes two scalars associated with losses - for discriminator
    # and generator 
    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        
        # argument passed - data identifier, scalar_name, global_step_mnumber
        self.writer.add_scalar('{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar('{}/G_error'.format(self.comment), g_error, step)
            
    # this function is meant to transform the images to the desired format  
    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):
        '''
        input images are expected in format (NCHW) Batch No., Channel, Height, Width
        '''
        # convert it to torch tensor
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        # get the desired image dimensions
        if format=='NHWC':
            images = images.transpose(1,3)
        

        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '_' + str(step))

        # Make horizontal grid from image tensor, normalization is MinMaxScaling
        # scale_each option if true would take min and max value for the individual image rather than 
        # the entire batch of images 
        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor, normalization is MinMaxScaling
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        # format would be tag_name, tensor to be displayed, global_step_number
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    
    # this function saves a given matplotlib figure passed using fig argument
    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        # finally save the images
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,comment, epoch, n_batch))
        
    # saving the two versions of the image grids 
    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            # get the current figure
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()
        
    # function to display the current status of the losses, prediction probs 
    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
        
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data
               
        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, n_batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))


    # saving the model state after each epoch
    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),'{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),'{}/D_epoch_{}'.format(out_dir, epoch))

    # function is for closing the writer
    def close(self):
        self.writer.close()
        
        
    # two important static methods
    # important note: static method is linked to the class directly and does not need self or cls 
    # argument
    # this function calculates the step number
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    # this function it creates directory per requirements
    # important note: raise keyword is used to reraise the current exception 
    # in an exception handler, so that it can be handled further up the call stack.
    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    