'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import os

def save_visualization_slices(x, y, x_def, count, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    slices = [1/3, 1/2, 2/3]

    # Creating a 3x3 grid for subfigures, with each row representing one of x, y, x_def
    fig, axs = plt.subplots(3, 3, figsize=(15, 15)) 

    for col, slice_frac in enumerate(slices):
        slice_idx = int(x.shape[2] * slice_frac)  # Adjusting slice index based on the fraction

        # Plotting slices for x
        x_slice = x[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[0, col].imshow(x_slice, cmap='gray')
        axs[0, col].set_title(f'X Slice {int(slice_frac*100)}%')
        
        # Plotting slices for y
        y_slice = y[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[1, col].imshow(y_slice, cmap='gray')
        axs[1, col].set_title(f'Y Slice {int(slice_frac*100)}%')

        # Plotting slices for x_def
        x_def_slice = x_def[0, 0, slice_idx, :, :].detach().cpu().numpy()
        axs[2, col].imshow(x_def_slice, cmap='gray')
        axs[2, col].set_title(f'X_def Slice {int(slice_frac*100)}%')

    # Turn off axis visibility for all subplots
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    # Save the figure with a unique name based on 'count'
    fig.savefig(os.path.join(results_dir, f"visual_{count}.png"))
    plt.close(fig)  # Close the figure to free memory

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VxmDense_1(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow
        
class VxmDense_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        self.unet_model2 = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
        self.flow2 = Conv(self.unet_model2.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.flow2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        self.transformer2 = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # first stage
        source = x[:, 0:1, :, :]
        source_y = x[:, 1:2, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        y_source = self.transformer(source, pos_flow)
        # save_visualization_slices(source, source_y, y_source, 1, results_dir='results')

        # second stage
        source2_x_stage1 = self.transformer(x2, pos_flow)
        x2_stage1 = torch.cat((source2_x_stage1,y2),dim=1)
        x2_stage1 = self.unet_model2(x2_stage1)
        flow_field2 = self.flow2(x2_stage1)
        pos_flow2 = flow_field2
        y_source2 = self.transformer2(source2_x_stage1, pos_flow2)
        # save_visualization_slices(x2, y2, y_source2, 1, results_dir='results_edge')

        # warp image with flow field
        y_source = self.transformer(source, pos_flow + pos_flow2)
        y_source2 = self.transformer2(x2, pos_flow + pos_flow2)
        # save_visualization_slices(source, source_y, y_source, 2, results_dir='results')
        # save_visualization_slices(x2, y2, y_source2, 2, results_dir='results_edge')

        return y_source, y_source2, (pos_flow + pos_flow2)

class VxmDense_huge(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=((14, 28, 144, 320), (1152, 1152, 320, 144, 28, 14, 14)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow