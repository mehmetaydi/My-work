
Image to Image translation with Generative Adversarial Networks (GAN).

- Please download Cycle GAN repo via this link [link](https://github.com/junyanz/CycleGAN).

- create a folder .\checkpoints\input2target_pretrained_model  and add  pre-trained model latest_net_G.pth into it as depicted below. 
![image](https://user-images.githubusercontent.com/101706254/196416786-f9274f25-aaba-453e-83ef-bc0ad88af2cb.png)


-   Create a dataset folder under `/dataset` for your dataset.
-   Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. Place any images you want to transform from a to b (input2target) in the `testA` folder, images you want to transform from b to a (input2target) in the `testB` folder, and do the same for the `trainA` and `trainB` folders as shown below.


![image](https://user-images.githubusercontent.com/101706254/196416659-f71b453a-0cab-4e9b-9f29-853da5b27972.png)

dataset i used for this project is available in [Visidon Oy](https://www.visidon.fi/) career [webpage](https://www.visidon.fi/careers/). 
 Take the first 200 images as test and the rest as train dataset.
