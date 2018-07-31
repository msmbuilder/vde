# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors: Muneeb Sultan <msultan@stanford.edu>
#               Hannah Wayment-Steele <h.wayment-steele@stanford.edu>
#               Brooke Husic <brookehusic@gmail.com>,
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from msmbuilder.base import BaseEstimator

from .utils import initialize_weights

__all__ = ['VDE']


def Layer(i, o, activation=None, p=0., bias=True):
    model = [nn.Linear(i, o, bias=bias)]
    if activation == 'SELU':
        model += [nn.SELU(inplace=True)]
    elif activation == 'RELU':
        model += [nn.ReLU(inplace=True)]
    elif activation == 'LeakyReLU':
        model += [nn.LeakyReLU(inplace=True)]
    elif activation == 'Sigmoid':
        model += [nn.Sigmoid()]
    elif activation == 'Tanh':
        model += [nn.Tanh()]
    elif activation == 'Swish':
        model += [Swish()]
    elif type(activation) is str:
        raise ValueError('{} activation not implemented.'.format(activation))

    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Encoder(nn.Module):
    """Encoder network for dimensionality reduction to latent space"""
    def __init__(self, input_size, output_size=1, hidden_layer_depth=5,
                 hidden_size=1024, activation='Swish', dropout_rate=0.):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = Layer(input_size, hidden_size,
                                 activation=activation, p=dropout_rate)
        net = [Layer(hidden_size, hidden_size, activation=activation,
                     p=dropout_rate) for _ in range(hidden_layer_depth)]
        self.hidden_network = nn.Sequential(*net)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out


class Lambda(nn.Module):
    """Application of Gaussian noise to the latent space"""
    def __init__(self, i=1, o=1, scale=1E-3):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class Decoder(nn.Module):
    """Decoder network for reconstruction from latent space"""
    def __init__(self, output_size, input_size=1, hidden_layer_depth=5,
                 hidden_size=1024, activation='Swish', dropout_rate=0.):
        super(Decoder, self).__init__()
        self.input_layer = Layer(input_size, input_size, activation=activation)

        net = [Layer(input_size, hidden_size,
                     activation=activation, p=dropout_rate)]
        net += [Layer(hidden_size, hidden_size, activation=activation,
                      p=dropout_rate) for _ in range(hidden_layer_depth)]

        self.hidden_network = nn.Sequential(*net)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out

class Model(nn.Module):
    """Full VDE model"""
    def __init__(self, input_size, encoder_size=1,
                 hidden_layer_depth=3, hidden_size=2048, scale=1E-3,
                 dropout_rate=0., activation='Swish', cuda=False):

        super(Model, self).__init__()
        
        self.encoder = Encoder(input_size, output_size=encoder_size,
                               hidden_layer_depth=hidden_layer_depth,
                               hidden_size=hidden_size, activation=activation,
                               dropout_rate=dropout_rate)
        self.lmbd = Lambda(encoder_size, encoder_size, scale=scale)
        self.decoder = Decoder(input_size, input_size=encoder_size,
                               hidden_layer_depth=hidden_layer_depth,
                               hidden_size=hidden_size, activation=activation,
                               dropout_rate=dropout_rate)
        
        self.dtype = torch.FloatTensor
        if cuda:
            self.cuda()
            self.dtype = torch.cuda.FloatTensor
        
        self.apply(initialize_weights)
        
    def forward(self, x):
        u = self.encoder(x)
        u_p = self.lmbd(u)
        out = self.decoder(u_p)
        return out, u

class VDE(BaseEstimator):
    """Variational Dynamical Encoder (VDE)

    Non-linear dimensionality reduction using a time-lagged variational
    autoencoder which projects data into a one-dimensional space.

    Parameters
    ----------
    lag_time : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+lag_time].
    batch_size : int, default=100
        Batch size to use during SGD optimization.
    hidden_layer_depth : int, default=3
        Number of hidden layers.
    scale : float, default=1E-3
        Noise scaling for the variational layer.
    dropout_rate : float, default=0.
        Dropout rate for hidden layers.
    learning_rate : float, default=1E-4
        Learning rate used for optimization.
    n_epochs : int, default=5
        Number of epochs to use during optimization.
    optimizer : str, default='Adam'
        Optimizer to use during SGD optimization. Choices include 'Adam' or
        'SGD'.
    activation : str, default='Swish'
        Non-linear activation function to use.
    loss : str, default='MSELoss'
        Reconstruction loss function to use.
    sliding_window : bool, default=True
        Whether or not to use a sliding window for training data augmentation.
    autocorr : bool, default=True
        Whether or not to use the autocorrelation loss.
    cuda : bool, default=False
        Whether or not to use CUDA.
    verbose : bool, default=True
        Print out loss information.
    """
    def __init__(self, input_size, lag_time=1, encoder_size=1, batch_size=100,
                 hidden_layer_depth=3, hidden_size=2048, scale=1E-3,
                 dropout_rate=0., learning_rate=1E-4, n_epochs=5,
                 optimizer='Adam', activation='Swish', loss='MSELoss',
                 sliding_window=False, autocorr=True, cuda=False,
                 verbose=True):

        super(VDE, self).__init__()

        self.input_size = input_size
        self.lag_time = lag_time
        self.encoder_size = encoder_size
        self.batch_size = batch_size
        self.hidden_layer_depth = hidden_layer_depth
        self.hidden_size = hidden_size
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.sliding_window = sliding_window
        self.autocorr = autocorr
        self.cuda = cuda
        self.verbose = verbose
        
        self._init_model()
        
        
    def _init_model(self):
        
        self._model = Model(self.input_size, encoder_size=self.encoder_size,
                            hidden_layer_depth=self.hidden_layer_depth, hidden_size=self.hidden_size,
                            scale=self.scale, dropout_rate=self.dropout_rate, activation=self.activation,
                            cuda=self.cuda)
        
        if optimizer == 'Adam':
            self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self._optimizer = optim.SGD(self._model.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=True)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=True)
        else:
            raise ValueError('Not a recognized loss function')

        self.is_fitted = False
        
        return self
    
    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self._init_model()

    def __repr__(self):
        return """VDE(input_size={input_size}, encoder_size={encoder_size}, n_epochs={n_epochs},
    batch_size={batch_size}, lag_time={lag_time}, sliding_window={sliding_window},
    autocorr={autocorr}, cuda={cuda})""".format(
            input_size=self.input_size,
            encoder_size=self.encoder_size,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lag_time=self.lag_time,
            sliding_window=self.sliding_window,
            autocorr=self.autocorr,
            cuda=self.cuda
        )

    def _rec(self, x_decoded_mean, x, loss_fn):
        z_mean, z_log_var = self._model.lmbd.mu, self._model.lmbd.log_v
        loss = loss_fn(x_decoded_mean, x)
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))
        return kl_loss + loss

    def _corr(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x.expand_as(x))
        ym = y.sub(mean_y.expand_as(y))
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        return r_val

    def compute_loss(self, X):
        x = Variable(X[:, :, 0].type(self._model.dtype), requires_grad=True)
        y = Variable(X[:, :, 1].type(self._model.dtype), requires_grad=True)

        o, u = self._model(x)

        autocorr_loss = 0.
        rec_loss = self._rec(o, y.detach(), self.loss_fn)

        loss = rec_loss
        if self.autocorr:
            v = self._model.encoder(y)
            autocorr_loss = (1 - self._corr(u, v))
            loss = rec_loss + autocorr_loss

        self._optimizer.zero_grad()
        loss.backward()
        return loss, rec_loss, autocorr_loss, x

    def _train(self, data, print_every=100):
        self._model.train()

        for t, X in enumerate(data):
            loss, rec_loss, autocorr_loss, _ = self.compute_loss(X)

            if (t + 1) % print_every == 0 and self.verbose:
                print('Batch %d, loss = %.4f' % (t + 1, loss.data[0]))

                if self.autocorr:
                    print('rec_loss = %.4f, '
                          'autocorr_loss = %.4f' % (rec_loss.data[0],
                                                    autocorr_loss.data[0]))
            self._optimizer.step()

    def _create_dataset(self, data):
        slide = self.lag_time if self.sliding_window else 1

        t0 = np.concatenate([d[j::self.lag_time][:-1] for d in data
                             for j in range(slide)], axis=0)
        t1 = np.concatenate([d[j::self.lag_time][1:] for d in data
                             for j in range(slide)], axis=0)
        t = np.concatenate((t0.reshape(-1, self.input_size, 1),
                            t1.reshape(-1, self.input_size, 1)), axis=-1)

        return DataLoader(t, batch_size=self.batch_size, shuffle=True,
                          drop_last=True)

    def fit(self, X):
        train_data = self._create_dataset(X)

        for i in range(self.n_epochs):
            if self.verbose:
                print('Epoch: %s' % i)
            self._train(train_data)

        self.is_fitted = True

    def _batch_transform(self, x):
        y = []
        for arr in np.array_split(x, x.shape[0] // self.batch_size):
            out = self._model.encoder(Variable(
                torch.from_numpy(arr).type(self._model.dtype))
            ).cpu().data.numpy()
            y.append(out.reshape(-1, self.encoder_size))

        return np.concatenate(y, axis=0)

    def propagate(self, X, scale=None):
        self._model.eval()
        if self.is_fitted:
            out = self._model.encoder(Variable(
                torch.from_numpy(X.reshape(-1, self.input_size)
                                 ).type(self._model.dtype)))
            if scale is not None:
                old_scale = self._model.lmbd.scale
                self._model.lmbd.scale = scale
                out = self._model.lmbd(out)
                self._model.lmbd.scale = old_scale
            return self._model.decoder(out).cpu().data.numpy()
        raise RuntimeError('Model needs to be fit.')

    def transform(self, X):
        self._model.eval()
        if self.is_fitted:
            out = [self._batch_transform(x) for x in X]
            return out
        raise RuntimeError('Model needs to be fit.')

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def compute_saliency(self, data, add_n_lag_zeros=True):
        self._model.eval()
        saliency_list = []
        scale = self._model.lmbd.scale
        self._model.lmbd.scale = 0.

        for t, X in enumerate(data):
            _, _, _, x0 = self.compute_loss(X)
            saliency = torch.abs(x0.grad.data)
            saliency = saliency.squeeze()
            saliency_list.append(saliency)

        self._model.lmbd.scale = scale
        if not add_n_lag_zeros:
            return np.vstack([i.numpy() for i in saliency_list])
        else:
            return np.vstack((np.vstack([i.numpy() for i in saliency_list]),
                              np.zeros((self.lag_time * self.n_lags,
                                        x0.size()[1]))))
