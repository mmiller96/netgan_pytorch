# NetGAN: Generating Graphs via Random Walks
**Pytorch implementation of the method proposed in the paper:**
[NetGAN: Generating Graphs via Random Walks](https://arxiv.org/abs/1803.00816)  
**based on the tensorflow implementation:**
https://github.com/danielzuegner/netgan  
The generator and the discriminator are defined in *models.py*. For better understanding the architectures of the models are shown as images below. The hyperparameters are defined respectively. *main.py* is a little demo version where a graph is created from a power grid.
# How GANs work:  
![GAN](https://user-images.githubusercontent.com/17961647/81090125-8eb02500-8efd-11ea-8df5-34ec4ad643f7.png)  
# Generator model:  
![Generator](https://user-images.githubusercontent.com/17961647/81085459-88b74580-8ef7-11ea-9614-368f8543a1f2.png)
# Discriminator model: 
![Discriminator](https://user-images.githubusercontent.com/17961647/81088760-bb633d00-8efb-11ea-8301-bd4887d91b91.png)
# Generator model expanded with conductor length
![Generator_expanded](https://user-images.githubusercontent.com/17961647/83631876-869be180-a59e-11ea-83d8-2ba005c09983.png)
