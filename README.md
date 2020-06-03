# NetGAN: Generating Graphs via Random Walks
**Pytorch implementation of the method proposed in the paper:**
[NetGAN: Generating Graphs via Random Walks](https://arxiv.org/abs/1803.00816)  
**based on the tensorflow implementation:**
https://github.com/danielzuegner/netgan  
There are two folders "netgan" and "netgan_modified". The first folder is the normal netgan implementation. It contains four different python files *training.py*, *models.py*, *utils.py* and *demo_pytorch.ipynb*. *training.py* is the main file. With this file you can train a graph and generate synthetic graphs afterwards. *models.py* contains the generator and the discriminator. *utils.py* has usefull functions and the *demo_pytorch.ipynb* is a demo version where training in done on the cora dataset. 
For better understanding the architectures of the models are shown as images below. The hyperparameters are defined respectively. 

The folder "netgan_modified" is a modifed version of netgan. The generator has changed as the bottom picture shows. With the additional LSTM it is possible to generate graphs with an additional feature. *demo_pytorch.ipynb* is an example where synthetic graphs are created from an electrical grid. The structure and the conduction length are generated. *branch.csv* and *bus.csv* contain information from different electircal grids. They are created from https://electricgrids.engr.tamu.edu/electric-grid-test-cases/
# How GANs work:  
![GAN](https://user-images.githubusercontent.com/17961647/81090125-8eb02500-8efd-11ea-8df5-34ec4ad643f7.png)  
# Generator model:  
![Generator](https://user-images.githubusercontent.com/17961647/81085459-88b74580-8ef7-11ea-9614-368f8543a1f2.png)
# Discriminator model: 
![Discriminator](https://user-images.githubusercontent.com/17961647/81088760-bb633d00-8efb-11ea-8301-bd4887d91b91.png)
# Generator model expanded with conductor length
![Generator_expanded](https://user-images.githubusercontent.com/17961647/83631876-869be180-a59e-11ea-83d8-2ba005c09983.png)
