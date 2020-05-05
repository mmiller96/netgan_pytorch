from training import Trainer
import utils
import pdb
import torch
import matplotlib.pyplot as plt
import numpy as np

create_graph_every = 500
path = r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\2000.pt'
graph = utils.get_graph(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\graphs\branch.csv', '2000')
trainer = Trainer(graph, 2000, max_iterations=40000, rw_len=12, batch_size=256, H_gen=40, H_disc=30, H_inp=128, z_dim=16, lr=0.0008,
                  n_critic=3, gp_weight=10.0, betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0, create_graph_every=create_graph_every, max_patience=5, stopping_criterion='val', stopping_eo=0.5,
                  val_share=0.1, test_share=0.05, num_samples_graph=50000, n_jobs=-1)
trainer.train()
torch.save(trainer, path)

plt.plot(trainer.critic_loss, label='Discriminator', color='b')
plt.plot(trainer.generator_loss, label='Generator', color='r')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.legend(loc="upper left")
plt.savefig(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\critic_generator_loss.pdf')
plt.close()

plt.plot(np.arange(len(trainer.eo))*create_graph_every, trainer.eo)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Edge overlaps')
plt.savefig(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\eo.pdf')
plt.close()

plt.plot(np.arange(len(trainer.roc_auc))*create_graph_every, trainer.roc_auc)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Roc Auc')
plt.savefig(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\roc_auc.pdf')
plt.close()

plt.plot(np.arange(len(trainer.avp))*create_graph_every, trainer.avp)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Average Precision Score')
plt.savefig(r'C:\Users\Data Miner\PycharmProjects\Master_Projekt4\avp.pdf')
plt.close()