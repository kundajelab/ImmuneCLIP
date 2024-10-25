'''
Contents of this file are from the open source code for

  NaderiAlizadeh, Navid, and Rohit Singh.
  Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport.
  bioRxiv (2024): 2024-01.

MIT License

Copyright (c) 2024 Navid NaderiAlizadeh and Rohit Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import contextlib

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

class SWE_Pooling(nn.Module):
    def __init__(self, d_in, num_ref_points, num_slices):
        '''
        Produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size based on sliced-Wasserstein embedding.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_slices: Number of slices
        '''
        super(SWE_Pooling, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_slices = num_slices

        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_slices)
        self.reference = nn.Parameter(uniform_ref)

        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_slices, bias=False), dim=0)
        if num_slices <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
            
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Linear(num_ref_points, 1, bias=False)

    def forward(self, X, mask=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
            mask [optional]: B x N binary tensor, with 1 iff the set element is valid; used for the case where set sizes are different
        Output:
            weighted_embeddings: B x num_slices tensor, containing a batch of B embeddings, each of dimension "num_slices" (i.e., number of slices)
        '''

        B, N, _ = X.shape       
        Xslices = self.get_slice(X)

        M, _ = self.reference.shape

        if mask is None:
            # serial implementation should be used if set sizes are different
            Xslices_sorted, Xind = torch.sort(Xslices, dim=1)

            if M == N:
                Xslices_sorted_interpolated = Xslices_sorted
            else:
                x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
                xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
                y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
                Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2)
        else:
            # replace invalid set elements with points to the right of the maximum element for each slice and each set (which will not impact the sorting and interpolation process)
            invalid_elements_mask = ~mask.bool().unsqueeze(-1).repeat(1, 1, self.num_slices)
            Xslices_copy = Xslices.clone()
            Xslices_copy[invalid_elements_mask] = -1e10

            top2_Xslices, _ = torch.topk(Xslices_copy, k=2, dim=1)
            max_Xslices = top2_Xslices[:, 0].unsqueeze(1)
            delta_y = - torch.diff(top2_Xslices, dim=1)

            Xslices_modified = Xslices.clone()

            Xslices_modified[invalid_elements_mask] = max_Xslices.repeat(1, N, 1)[invalid_elements_mask]

            delta_x = 1 / (1 + torch.sum(mask, dim=1, keepdim=True))
            slope = delta_y / delta_x.unsqueeze(-1).repeat(1, 1, self.num_slices) # B x 1 x num_slices
            slope = slope.repeat(1, N, 1)

            eps = 1e-3
            x_shifts = eps * torch.cumsum(invalid_elements_mask, dim=1)
            y_shifts = slope * x_shifts
            Xslices_modified = Xslices_modified + y_shifts

            Xslices_sorted, _ = torch.sort(Xslices_modified, dim=1)

            x = torch.arange(1, N + 1).to(X.device) / (1 + torch.sum(mask, dim=1, keepdim=True)) # B x N

            invalid_elements_mask = ~mask.bool()
            x_copy = x.clone()
            x_copy[invalid_elements_mask] = -1e10
            max_x, _ = torch.max(x_copy, dim=1, keepdim=True)
            x[invalid_elements_mask] = max_x.repeat(1, N)[invalid_elements_mask]

            x = x.unsqueeze(1).repeat(1, self.num_slices, 1) + torch.transpose(x_shifts, 1, 2)
            x = x.view(-1, N) # BL x N

            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
            Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2)

        Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1) # B x num_slices x M

        weighted_embeddings = self.weight(embeddings).sum(-1)

        return weighted_embeddings.view(-1, self.num_slices)

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)