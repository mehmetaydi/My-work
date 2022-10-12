
Image to Image translation with Generative Adversarial Networks (GAN).

- Please download Cycle GAN repo via this link [link](https://github.com/junyanz/CycleGAN).

- create a folder .\checkpoints\input2target_pretrained_model  and add  pre-trained model latest_net_G.pth into it. 

-   Create a dataset folder under `/dataset` for your dataset.
-   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (input2target) in the `testA` folder, images you want to transform from b to a (input2target) in the `testB` folder, and do the same for the `trainA` and `trainB` folders as shown below.


image.png