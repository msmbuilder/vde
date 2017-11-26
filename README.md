Variational Dynamical Encoder (VDE)
===

Often the analysis of time-dependent chemical and biophysical systems produces high-dimensional time-series data for which it can be difficult to interpret which features are most salient in defining the observed dynamics. While recent work from our group and others has demonstrated the utility of time-lagged co-variate models to study such systems, linearity assumptions can limit the compression of inherently nonlinear dynamics into just a few characteristic components. Recent work in the field of deep learning has led to the development of variational autoencoders (VAE), which are able to compress complex datasets into simpler manifolds. We present the use of a time-lagged VAE, or variational dynamics encoder (VDE), to reduce complex, nonlinear processes to a single embedding with high fidelity to the underlying dynamics. We demonstrate how the VDE is able to capture nontrivial dynamics in a variety of examples, including Brownian dynamics and atomistic protein folding. Additionally, we demonstrate a method for analyzing the VDE model, inspired by saliency mapping, to determine what features are selected by the VDE model to describe dynamics. The VDE presents an important step in applying techniques from deep learning to more accurately model and interpret complex biophysics.

Requirements
---
+ ``numpy``
+ ``pytorch``
+ ``msmbuilder``

Usage
---

Using the VDE is as easy as using any [`msmbuilder`](http://www.msmbuilder.org) model:

```python
from vde import VDE
from msmbuilder.example_datasets import MullerPotential

trajs = MullerPotential().get().trajectories

lag_time = 10
vde_mdl = VDE(2, lag_time=lag_time, hidden_layer_depth=3,
          sliding_window=True, cuda=True, n_epochs=10,
          learning_rate=5E-4)

latent_output = vde_mdl.fit_transform(trajs)
```

Cite
---
If you use this code in your work, please cite:

```bibtex
@article{Hernandez2017,
  title={Variational Encoding of Complex Dynamics},
  author={Hern\'{a}ndez, Carlos X. and Wayment-Steele, Hannah K. and Sultan, Mohammad M. and Husic, Brooke E. and Pande, Vijay S.},
  journal={arXiv preprint},
  year={2017}
}

```
