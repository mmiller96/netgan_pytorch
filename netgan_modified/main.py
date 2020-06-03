from training import Trainer
import utils
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    _A_obs, _X_obs, _z_obs = utils.load_npz('cora_ml.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]
    get_graph()
    trainer = Trainer(_A_obs, _N, max_iterations=40000, rw_len=16, batch_size=128, H_gen=40, H_disc=30, H_inp=128, z_dim=16, lr=0.0003,
                      n_critic=3, gp_weight=10.0, betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0,
                      val_share=0.1, test_share=0.05, seed=0)

    create_every = 500
    plot_every = 100

    trainer.train(create_graph_every=create_every, plot_graph_every=plot_every, num_samples_graph=500000, stopping_criterion='val')