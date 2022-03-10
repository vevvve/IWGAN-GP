import tensorflow as tf 
import os
import sys 
sys.path.insert(0, '../')
import tensorlayer as tl
import numpy as np
import random 
from glob import glob
import argparse
import scripts
from scripts.GANutils import *
from scripts.models import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser(description='3D-GAN implementation for 32*32*32 voxel output')
parser.add_argument('-n','--name', default='Test', help='The name of the current experiment, this will be used to create folders and save models.')
parser.add_argument('-d','--data', default='./data', help ='The location fo the object voxel models.' )
parser.add_argument('-e','--epochs', default=20001, help ='The number of epochs to run for.', type=int)
parser.add_argument('-b','--batchsize', default=2, help ='The batch size.', type=int)
parser.add_argument('-sample', default= 20000, help='How often generated obejcts are sampled and saved.', type= int)
parser.add_argument('-save', default= 50000, help='How often the network models are saved.', type= int)
parser.add_argument('-l', '--load', default= False, help='Indicates if a previously loaded model should be loaded.', action = 'store_true')
parser.add_argument('-le', '--load_epoch', default= '', help='The epoch to number to be loaded from.', type=str)
parser.add_argument('-graph', default= 20000, help='How often the discriminator loss and the reconstruction loss graphs are saved.', type= int)
args = parser.parse_args()

checkpoint_dir = "checkpoint/" + args.name +'/'
save_dir =  "savepoint/" + args.name +'/'
output_size = 64


######### make directories ################
make_directories(checkpoint_dir,save_dir)

####### inputs  ###################
z = tf.random_normal((args.batchsize, 200), 0, 1)  #z 200
real_models =  tf.placeholder(tf.float32, [args.batchsize, output_size, output_size, output_size] , name='real_models')
########## network computations #######################



#used for training genorator
net_g, G_Fake =  generator_32(z, is_train=True, reuse = False, sig= False, batch_size=args.batchsize)


#used for training d on fake
net_d, D_Fake  = discriminator(G_Fake, output_size, batch_size= args.batchsize, improved = True ,is_train = True, reuse= False)
#used for training d on real
net_d2, D_Real = discriminator(real_models, output_size, batch_size= args.batchsize, improved = True ,is_train = True, reuse= True)

########### Loss calculations ############

alpha               = tf.random_uniform(shape=[args.batchsize,1] ,minval =0., maxval=1.) # here we calculate the gradient penalty 
difference          = G_Fake - real_models
inter               = []
for i in range(args.batchsize): 
    inter.append(difference[i] *alpha[i])
inter = tf.stack(inter)
interpolates        = real_models + inter
gradients           = tf.gradients(discriminator(interpolates, output_size, batch_size= args.batchsize, improved = True, is_train = False, reuse= True)[1],[interpolates])[0]
slopes              = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
gradient_penalty    = tf.reduce_mean((slopes-1.)**2.)
    
d_loss = -tf.reduce_mean(D_Real) + tf.reduce_mean(D_Fake) + 10.*gradient_penalty
g_loss = -tf.reduce_mean(D_Fake)

############ Optimization #############


g_vars = tl.layers.get_variables_with_name('gen', True, True)   
d_vars = tl.layers.get_variables_with_name('dis', True, True)


net_g.print_params(False)
net_d.print_params(False)

d_optim = tf.train.AdamOptimizer( learning_rate = 2e-4, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer( learning_rate = 2e-4, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)


####### Training ################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.Session(config=config)
#sess=tf.Session()
#tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
sess.run(tf.global_variables_initializer())

# load checkpoints
if args.load: 
    load_networks(checkpoint_dir, sess, net_g, net_d, epoch = args.load_epoch)
    track_d_loss_iter, track_d_loss,_ = load_values(save_dir)
else:
    track_d_loss_iter, track_d_loss, iter_counter,track_g_loss, track_g_loss_iter = [],[],0,[],[]

iter_counter = iter_counter - (iter_counter %5)
files,_ = grab_files(args.data)
#training starts here  
for epoch in range(args.epochs):
    random.shuffle(files)
    # print(int(len(files)/args.batchsize))
    for idx in range(0, int(len(files)/args.batchsize)):
        file_batch = files[idx*args.batchsize:(idx+1)*args.batchsize]
        models, start_time = make_inputs(file_batch)
        # print(models.shape)
        if models.shape != (2,64,64,64):
            continue
        # updates the discriminator
        errD,_= sess.run([d_loss, d_optim] , feed_dict={ real_models: models }) 
        track_d_loss.append(-errD)
        track_d_loss_iter.append(iter_counter)
        
        # update the generator 
        if iter_counter % 5 ==0 :
            errG, _, objects= sess.run([g_loss, g_optim, G_Fake], feed_dict={})
            track_g_loss_iter.append(iter_counter)
            track_g_loss.append(errG)
        
        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, args.epochs, idx, len(files)/args.batchsize, time.time() - start_time, errD, errG))
        sys.stdout.flush()

        iter_counter += 1

    #saving generated objects
    if np.mod(epoch, args.sample ) == 0:     
        save_voxels(save_dir,objects, epoch)
     #saving the model 
    if np.mod(epoch, args.save) == 0:
        save_networks(checkpoint_dir,sess, net_g, net_d, epoch)
      

    #saving learning info 
    if np.mod(epoch, args.graph) == 0: 
        render_graphs(save_dir,epoch, track_d_loss_iter, track_d_loss,track_g_loss_iter,track_g_loss) #this will only work after a 50 iterations to allows for proper averating
        save_values(save_dir,track_d_loss_iter, track_d_loss) # same here but for 300


    
