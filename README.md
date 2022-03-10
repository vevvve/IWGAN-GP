Python implementation code for the paper titled,

Title: A 3D shale reconstruction method based on improved WGAN-GP

Authors: Ting Zhang1, Qingyang Liu1, Xianwu Wang1, Xin Ji1, Yi Du2, * 

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China 

2.College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

(*corresponding author, E-mail: duyi0701@126.com. Tel.: 86 - 21- 50214252. Fax: 86 - 21- 50214252. )

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Qingyang Liu: ve.vvve@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Xianwu Wang Email: wxw938491904@dingtalk.com, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Xin Ji Email: shiep_sky@163.com, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yi Du E-mail: duyi0701@126.com, Affiliation: College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

# IWGAN-GP


1.requirements

Tensorflow-gpu == 1.13.1

tensorlayer == 1.10.1

To run the code, an NVIDIA GeForce RTX3080 GPU video card with 10GB video memory is required. 

Software development environment should be any Python integrated development environment used on an NVIDIA video card. 

Programming language: Python 3.6. 


2.How to use？


First, preprocess the image: Cut the shale slice into 64 * 64 * 64 size pictures. Each training image consists of 64 pictures of size 64 * 64, stored in a separate folder. Then use use scripts/preparedata.py to convert the image into .npy format.


Secondly, set the network parameters such as batchsize, learning rate and storage location. The executable .py file of IWGAN-GP, the path is: IWGAN-GP/IWGAN-GP.py. After configuring the parameters and environment, you can run directly: python IWGAN-GP.py


Finally, in IWGAN-GP/savepoint/Test, find the loss images during the training process and the .npy format of the shale three-dimensional structure images of different rounds. Use scripts/loadnpy to convert .npy to .txt format for later analysis and processing.

