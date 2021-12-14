import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from PIL import Image, ImageDraw, ImageFont

import tfutil
import utils
import dataset




  
def discriminator_value_statistics(model, global_seed, n_statistics, gpu_id, experiment_id , optional_run_id, tfrecord_dir):

  print("Calculating statistics of critic values for model  " + model)
  print("")
    
  batch_size= 10
  n_batches = int(np.ceil(n_statistics/batch_size))
  
  # Lists and empty np arrays for collecting data
  
  max_values = []
  min_values = []
  critics = []
    
  real_scores_all = np.empty((batch_size,0),float) 
  fake_scores_all = np.empty((batch_size,0),float)

  # Initialize TensorFlow session.
  sess = tf.InteractiveSession()

  # Import official CelebA-HQ networks.

  G, D, Gs = utils.prepare_GAN_nets(sess, model)
  
  # Initialize data dictionary
  data = EasyDict(tfrecord_dir=tfrecord_dir) 
  training_set = dataset.load_dataset(verbose=False, **data)

  # Set up tensorflow graph
  latents = np.random.RandomState(global_seed).randn(n_statistics, *Gs.input_shapes[0][1:] )
  

  latents_plchldr = G.input_templates[0]
  labels_plchldr = G.input_templates[1]
  fakes = G.get_output_for( latents_plchldr, labels_plchldr, is_training=False ) # always (:,3,1024,1024)  
  
  disc_fakes, _ = utils.fp32( D.get_output_for(fakes, is_training=False ) )



  # Preprocessing real data information
  drange_in = [0,255]
  drange_out = [-1,1]
  scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
  bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
  
  
  
  print("\nCollect values:")
  
  for i in range(0,n_batches):
    percent = int(float(i+1)/n_batches * 100)
  

    # Get real samples and their discriminator values
    imgs,labels  = training_set.get_minibatch_np(minibatch_size=batch_size, lod=0)
    imgs = imgs * scale + bias
    real_scores_batch = D.run( imgs )[0]
      

    # Get fake samples and their discriminator values    
    latents_batch = latents[i*batch_size : min((i+1)*batch_size,n_statistics)]
    lbls_batch = np.zeros([latents_batch.shape[0]] + Gs.input_shapes[1][1:] )
    [fake_scores_batch] =  tf.get_default_session().run([disc_fakes], feed_dict={latents_plchldr : latents_batch, labels_plchldr :lbls_batch})
    
    fake_scores_batch = [item for sublist in fake_scores_batch for item in sublist] 
    
    # Collect data from this batch
    maxim_batch= np.max(fake_scores_batch)
    minim_batch= np.min(fake_scores_batch)
    max_values.append(maxim_batch)
    min_values.append(minim_batch)

    real_scores_all = np.append(real_scores_all, real_scores_batch)
    fake_scores_all = np.append(fake_scores_all, fake_scores_batch)

    print("Status: " + str(percent) + ' %', end="\r" )
  
  
  ######################################
  ## Find maximal and minimal values
  ######################################
  
  
  maxim = np.ceil(np.max(max_values))
  minim = np.floor(np.min(min_values))
  
  print("\nMaximal critic value found is  "+ str(maxim))
  print("Minimal critic value found is  "+ str(minim))
  print("Mean value is " + str(np.mean(fake_scores_all)))
  print("Standard deviation is " + str(np.std(fake_scores_all)))
  print("----------------------------")
  
  np.save("results/"+experiment_id+"/statistics/max_min.npy",[maxim,minim])
  
  ######################################  
  ## Plot histrogramm of critic values
  ######################################
  
  bins = np.linspace(minim, maxim, 50)

  plt.hist(real_scores_all, bins, alpha=0.5, label='reals')
  plt.hist(fake_scores_all, bins, alpha=0.5, label='fakes')
  #plt.xlim(right=300)
  plt.legend(loc='upper right', prop={'size': 24})
  plt.xticks(fontsize=18) 
  plt.yticks([], [])
  plt.savefig("results/"+experiment_id+"/statistics/histogram_" + optional_run_id + ".jpg",bbox_inches='tight')
  plt.close()

  print("")
  
  ######################################
  ## Sample grids:  
  ######################################

  print("Creating sample grids...") 
  
  ## Configs
  
  n_rows = 6
  n_per_row = 15
  n_grid = n_rows*n_per_row 
  dataset_dims= fakes.get_shape().as_list()[1:]
  
  fontsFolder = "Font/"
  arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'FreeSerif.ttf'), int(120/1024*dataset_dims[-1]))
  
  imheight = int((1024+200)/1024*dataset_dims[-1])
  imwidth = dataset_dims[-1]
  
  ## Sorting by critic values
    
  indexing_fake = np.argsort(fake_scores_all) 
       
  # apply the same ordering to the latents
  latents = np.take(latents,indexing_fake,axis=0)
      
  indices = {'low': np.arange(n_grid), 'random': np.sort(random.sample(range(n_statistics), n_grid)), 'high':np.arange(n_statistics-n_grid, n_statistics)}
  
  ##############################
  # Sample grid of real images
  print("\nSample grid of real samples")
  
  dst = Image.new('RGB', (imwidth * n_per_row, imheight* n_rows))
  
  # loop through minibatches to collect 100 images
  # 2 numpy arrays, one for images and one for disciminator values 
    
  real_images = np.empty((0,imwidth,imwidth,3), dtype=np.uint8)
  real_critics = np.empty(0)
  
  for i in range(n_rows):

    imgs,labels  = training_set.get_minibatch_np(minibatch_size=n_per_row, lod=0)
    imgs = imgs * scale + bias
    critics = D.run( imgs )[0]
    critics=np.array([x[0] for x in critics])


    imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
    imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

    real_images = np.append(real_images,imgsPlot,axis=0)
    real_critics = np.append(real_critics, critics)
    
    percent = int(100.0/n_rows*(i+1))
    print("Status: " + str(percent) + ' %', end="\r" )
    
  # sort discriminator values and remember the sorting operation

  indexing_real = np.argsort(real_critics) 
  real_critics = np.take(real_critics,indexing_real)
    
  # apply the same ordering to the images
  real_images = np.take(real_images,indexing_real,axis=0)
  
  # then loop through rows and columns and paste images
    
  # (i,j) is (column,row)  
  for i in range(n_rows): # loop over rows last
  
    for j in range(n_per_row): # loop over columns first
      img_plot = Image.fromarray( real_images[n_per_row*i+j], 'RGB' )         
      dst.paste(img_plot, (imwidth*j, imheight*i))
        
      header = Image.new('RGB', (imwidth, int(200/1024*imwidth)), 'white')
      draw = ImageDraw.Draw(header)
      draw.text( (int(200/1024*imwidth),int(20/1024*imwidth)), str(int(real_critics[i*n_per_row+j])), fill='black', font=arialFont)
      dst.paste(header, (imwidth*j,imheight*i+imwidth))
  
    
  if imwidth == 1024:
    size = int(dst.width/4),int(dst.height/4)
    dst_small = dst.resize(size,resample=Image.BILINEAR)    
  else:
    dst_small = dst
  
    
  dst.save("results/"+experiment_id+"/statistics/sample_grid_reals"+optional_run_id+".jpg") 
    
  print("")  
  ###########################
    
  ##############################
  # Sample grids of generated samples


  for clss,idx in indices.items():
    
    print("\nSample grid of generated samples with "+clss+" critic values")
  
    dst = Image.new('RGB', (imwidth * n_per_row, imheight* n_rows))
    
    latents_grid = np.take(latents,idx,axis=0)
    
    for i in range(n_rows):
      
      latents_row = latents_grid[i*n_per_row : (i+1)*n_per_row,:]
      labels_batch = np.zeros([n_per_row] + Gs.input_shapes[1][1:] )
        
      fake_samples, critics_samples =  sess.run([fakes,disc_fakes], feed_dict={latents_plchldr : latents_row, labels_plchldr :labels_batch})
            
           
      fake_samples = np.clip( np.rint( (fake_samples + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
      fake_samples = fake_samples.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
        
      # loop through rows and columns and paste images
      
      for j in range(n_per_row):
        img_plot = Image.fromarray( fake_samples[j], 'RGB' )         
        dst.paste(img_plot, (imwidth*j, imheight*i))
        
        header = Image.new('RGB', (imwidth, int(200/1024*imwidth)), 'white')
        draw = ImageDraw.Draw(header)
        draw.text( (int(200/1024*imwidth),int(20/1024*imwidth)), str(int(critics_samples[j])), fill='black', font=arialFont)
        dst.paste(header, (imwidth*j,imheight*i+imwidth))
      
      percent = int(100.0/n_rows*(i+1))
      print("Status: " + str(percent) + ' %', end="\r" )

        
    size = int(dst.width/4),int(dst.height/4)
        
    if imwidth == 1024:
      dst_small = dst.resize(size,resample=Image.BILINEAR)    
    else:
      dst_small = dst
        
    dst_small.save( "results/"+experiment_id+"/statistics/sample_grid_fakes_" + str(clss) +'_'+optional_run_id+".jpg") 
        
      
  print("\nAll sample grids collected.")
  
  print("\n---------------------\n")
  sess.close()
  tf.reset_default_graph()
  print
    
  return [maxim, minim]
  



################################################################
################################################################








def collect_local_variations(seed,model,experiment_id, n_samples = 10, n_perturbations = 10, epsilon=0.03):

    
    print("Collecting local perturbations...")
    
    # Initialize TensorFlow session.
    sess = tf.InteractiveSession()


    # Graph
    G, D, Gs = utils.prepare_GAN_nets(sess, model)

    labels_batch_plchldr = G.input_templates[1]
    latents_batch_plchldr = G.input_templates[0]
    samples_graph = G.get_output_for( latents_batch_plchldr, labels_batch_plchldr, is_training=False ) # celeba (:,3,1024,1024)
    critics_graph,_ = utils.fp32( D.get_output_for( samples_graph, is_training=False ) )


    
    latent_samples = np.random.RandomState( seed ).randn( n_samples, *Gs.input_shapes[0][1:] ) 
    labels_samples = np.zeros( [n_samples] + Gs.input_shapes[1][1:] )

    critic_samples =  tf.get_default_session().run(critics_graph, feed_dict={latents_batch_plchldr : latent_samples, labels_batch_plchldr :labels_samples })
    critic_samples=np.array([x[0] for x in critic_samples])
    
    indexing = np.argsort(critic_samples) 
    critic_samples= np.take(critic_samples,indexing)
    # apply the same ordering to the latents
    latent_samples = np.take(latent_samples,indexing,axis=0)

    
    latents = np.empty((0, *Gs.input_shapes[0][1:]),dtype=float)
    critics = []
    
    for i in range(0,n_samples):
        if i % 1 == 0:
            print( "Status: " + str( int( i/n_samples*100.0 )) + ' %', end="\r" )
        
            
        perturbations = np.random.RandomState( seed+100+i ).randn( n_perturbations,*Gs.input_shapes[0][1:])
        tempnorm = np.linalg.norm(perturbations,ord=2,axis=1,keepdims=True)
        #print(tempnorm)
        perturbation_norm = np.repeat(tempnorm,512,axis=1)
        perturbations = np.divide(perturbations,perturbation_norm)*np.sqrt(512)*epsilon
        
        latent_i_broadcast = np.repeat([latent_samples[i,:]],n_perturbations,axis=0)
        
        latents_perturbed = np.add(latent_i_broadcast,perturbations) 
        #print(latents_perturbed[:,0])
        labels_perturbed = np.zeros( [n_perturbations] + Gs.input_shapes[1][1:] )

        critic_perturbed =  tf.get_default_session().run(critics_graph, feed_dict={latents_batch_plchldr : latents_perturbed, labels_batch_plchldr :labels_perturbed })
        critic_perturbed=np.array([x[0] for x in critic_perturbed])
        #print(critic_perturbed)

        indexing = np.argsort(critic_perturbed) 
        critic_perturbed= np.take(critic_perturbed,indexing)
        latents_perturbed = np.take(latents_perturbed,indexing,axis=0)
        #print(np.shape(latents_perturbed))

        #print(np.shape(latents))
        critics = np.append(critics, critic_perturbed, axis=0)
        latents = np.append(latents, latents_perturbed, axis=0)
        #print(np.shape(latents))

    #print(latents[:,0])
        
    print("...Done!")        
    
    sess.close()
    tf.reset_default_graph()

        
    # return sorted critics
    return latents,critics


def collect_extreme_local_variations(seed,model,experiment_id, n_samples = 10, n_perturbations = 11, epsilon=0.03):

    if n_perturbations % 2 == 0:
        n_perturbations = n_perturbations + 1
    
    print("Collecting local perturbations...")
    
    # Initialize TensorFlow session.
    sess = tf.InteractiveSession()


    # Graph
    G, D, Gs = utils.prepare_GAN_nets(sess, model)

    labels_batch_plchldr = G.input_templates[1]
    latents_batch_plchldr = G.input_templates[0]
    samples_graph = G.get_output_for( latents_batch_plchldr, labels_batch_plchldr, is_training=False ) # celeba (:,3,1024,1024)
    critics_graph,_ = utils.fp32( D.get_output_for( samples_graph, is_training=False ) )


    
    latent_samples = np.random.RandomState( seed ).randn( n_samples, *Gs.input_shapes[0][1:] ) 
    labels_samples = np.zeros( [n_samples] + Gs.input_shapes[1][1:] )

    critic_samples =  tf.get_default_session().run(critics_graph, feed_dict={latents_batch_plchldr : latent_samples, labels_batch_plchldr :labels_samples })
    critic_samples=np.array([x[0] for x in critic_samples])
    
    indexing = np.argsort(critic_samples) 
    critic_samples= np.take(critic_samples,indexing)
    # apply the same ordering to the latents
    latent_samples = np.take(latent_samples,indexing,axis=0)

    
    n_huge = 960
    n_in_batch = 48
    n_batches = int(n_huge/n_in_batch)
    n_huge = n_in_batch*n_batches

    latents = np.empty((0, *Gs.input_shapes[0][1:]),dtype=float)
    critics = []
    
    for i in range(0,n_samples):
        #if i % 1 == 0:
            #print( "Status: " + str( int( i/n_samples*100.0 )) + ' %', end="\r" )
        
            
        perturbations = np.random.RandomState( seed+100+i ).randn( n_huge,*Gs.input_shapes[0][1:])
        tempnorm = np.linalg.norm(perturbations,ord=2,axis=1,keepdims=True)
        #print(tempnorm)
        perturbation_norm = np.repeat(tempnorm,512,axis=1)
        perturbations = np.divide(perturbations,perturbation_norm)*np.sqrt(512)
        
        latent_i_broadcast = np.repeat([latent_samples[i,:]],n_huge,axis=0)
        
        latents_perturbed = np.add(latent_i_broadcast,perturbations*epsilon) 
        #print(latents_perturbed[:,0])
        labels_perturbed = np.zeros( [n_huge] + Gs.input_shapes[1][1:] )
        labels_temp = np.zeros( [n_in_batch] + Gs.input_shapes[1][1:] )

        critic_perturbed = []
       
        for j in range(n_batches): 
            print( "Status: " + str( int( (i*n_batches+j)/(n_samples*n_batches)*100.0 )) + ' %', end="\r" )
            latents_temp = latents_perturbed[j*n_in_batch:(j+1)*n_in_batch,:]
            critic_temp =  tf.get_default_session().run(critics_graph, feed_dict={latents_batch_plchldr : latents_temp, labels_batch_plchldr :labels_temp })
            critic_temp = np.array([x[0] for x in critic_temp])
            critic_perturbed = np.append(critic_perturbed,critic_temp)
        #print(critic_perturbed)

        indexing = np.argsort(critic_perturbed)
        critic_perturbed= np.take(critic_perturbed,indexing)
        latents_perturbed = np.take(latents_perturbed,indexing,axis=0)
        #print(np.shape(latents_perturbed))

        n_on_each_side = int(n_perturbations/2)
        indices_low = range(n_on_each_side)
        indices_high = range(n_huge-n_on_each_side,n_huge)

        critic_low = np.take(critic_perturbed,indices_low)
        latents_low = np.take(latents_perturbed,indices_low,axis=0)
        critic_high = np.take(critic_perturbed,indices_high)
        latents_high = np.take(latents_perturbed,indices_high,axis=0)
        
        critic_perturbed = np.append(critic_low,critic_samples[i])
        critic_perturbed = np.append(critic_perturbed,critic_high)

        #print(np.shape(latents_low))
        #print(np.shape([latent_samples[i,:]]))
        #print(np.shape(latents_high))
        latents_perturbed = np.append(latents_low,[latent_samples[i,:]],axis=0)
        latents_perturbed = np.append(latents_perturbed,latents_high,axis=0)
        #print(np.shape(latents_perturbed))
        #print(np.shape(latents))
        critics = np.append(critics, critic_perturbed, axis=0)
        latents = np.append(latents, latents_perturbed, axis=0)
        #print(np.shape(latents))

    #print(latents[:,0])
        
    print("...Done!")        
    
    sess.close()
    tf.reset_default_graph()

        
    # return sorted critics
    return latents,critics

def plot_local_variation_sample_grids(model, gpu_id, experiment_id, n_samples = 10, seed = 1, epsilon = 0.03):

    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    #Initialize TensorFlow session.
    

    # Sample n_collection latent points randomly and order them. 
    # for every point, 
        #create 10 small pertubations of that latent point.
    n_perturbations = 15
        #calculate critic and sort 
    # aggregate into 10x10 list of latents and critics
    #latents,critics = collect_local_variations(seed,model,experiment_id, n_samples = n_samples, n_perturbations = n_perturbations, epsilon=epsilon)
    latents,critics = collect_extreme_local_variations(seed,model,experiment_id, n_samples = n_samples, n_perturbations = n_perturbations, epsilon=epsilon)
    # generate images and plot below:
    

    sess = tf.InteractiveSession()

    #Initialize data dictionary
    #data = EasyDict(tfrecord_dir=tfrecord_dir) 
    #training_set = dataset.load_dataset(verbose=False, **data)

    G, D, Gs = utils.prepare_GAN_nets(sess, model)

    

    # Graph:
    latents_plchldr = G.input_templates[0]
    labels_plchldr = G.input_templates[1]
    
    samples_graph = G.get_output_for( latents_plchldr, labels_plchldr, is_training=False ) # celeba (:,3,1024,1024)
    critics_graph,_ = utils.fp32( D.get_output_for( samples_graph, is_training=False ) )
        #print("Graph output: " + str(samples_graph))
    dataset_dims= samples_graph.get_shape().as_list()[1:]
  
    fontsFolder = "Font/"
    arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'FreeSerif.ttf'), int(120/1024*dataset_dims[-1]))


    samples = samples = np.empty((0,*G.output_shapes[0][1:]),dtype=np.uint8)
    critics = []
    print("Creating grid...")

    for i in range(n_samples):
        percent = int(i/n_samples*100)
        print("Status: " + str(percent) + ' %', end="\r" )

        #Get fake_scores, images (and gradients)
        latents_batch = latents[i*n_perturbations : (i+1)*n_perturbations,:]
        #print("latents shape: " + str(np.shape(latents_batch)))
        labels_batch = np.zeros([n_perturbations] + Gs.input_shapes[1][1:] )
        #print(np.shape(labels_batch))
        samples_batch,critics_batch =  tf.get_default_session().run([samples_graph,critics_graph], feed_dict={latents_plchldr : latents_batch, labels_plchldr :labels_batch})
        #print("after session: " + str(np.shape(samples_batch)))
        samples_batch=np.array([x for x in samples_batch])

        critics_batch=np.array([x[0] for x in critics_batch])
        #samples_batch=samples_batch[0,:,:,:,:]
        
        
        #rslt=np.array([x[0] for x in rslt])

        #print("to be appended: " +str(np.shape(samples_batch)))
        #print("appended to: " + str(np.shape(samples)))
        
        samples = np.append(samples, samples_batch, axis=0)
        critics = np.append(critics, critics_batch, axis=0)
        
        #print(np.shape(samples))
        
    samples = np.clip( np.rint( (samples + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
    samples = samples.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

    #print(np.shape(samples))

    imheight = int((1024+200)/1024*dataset_dims[-1])
    imwidth = dataset_dims[-1]
    #then loop through rows and columns and paste images
    dst = Image.new('RGB', (imwidth * n_perturbations, imheight * n_samples))

    for j in range(n_samples):
  
        for i in range(n_perturbations):
            #print(str(j)+","+str(i))
            img_plot = Image.fromarray( samples[j*n_perturbations+i], 'RGB' )         
            dst.paste(img_plot, (imwidth*i, imheight*j))
    
            header = Image.new('RGB', (imwidth, int(200/1024*imwidth)), 'white')
            draw = ImageDraw.Draw(header)
            if i == int(n_perturbations/2):
                draw.text((int(200/1024*imwidth),int(20/1024*imwidth)), str(critics[j*n_perturbations+i]), fill='red', font=arialFont)
            else:
                draw.text((int(200/1024*imwidth),int(20/1024*imwidth)), str(critics[j*n_perturbations+i]), fill='black', font=arialFont)
            dst.paste(header, (imwidth*i,imheight*j+imwidth))
    
    size = int(dst.width/4),int(dst.height/4)
    if imwidth == 1024:
        dst_small = dst.resize(size,resample=Image.BILINEAR)    
    else:
        dst_small = dst

    dst_small.save( "results/"+experiment_id+"/statistics/local_extreme_perturbations_grid.jpg") 



    sess.close()
    tf.reset_default_graph()

    print("\nDone\n------------------------\n")

    return None




## Convenience class from Nvidia
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]
    
    




