import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os
import pickle
import cv2
import glob
from functools import reduce

import tfutil 


from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras as keras



##########################################################################
##########################################################################


def plot_geodesic_comparison(geodesics_dict, methods, n_pts_on_geodesic, experiment_id, optional_run_id):

  imgs,_ ,_ = geodesics_dict[methods[0]]
  dims = list(imgs.shape[1:])
  
  if dims[0] == 3:
     color = "RGB"
  else:
     color = "L"

  dst = Image.new(color, (int(600/1024*dims[1])+dims[1] * n_pts_on_geodesic, dims[1] * len(methods)))
	
  fontsFolder = "Font/"
  textsize=int(400/1024*dims[1])
   
    
  for k_method in range(len(methods)):

    method = methods[k_method]
    text=""
    if method=="linear":
      text="(a)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )

    elif method=="sqDiff":
      text="(b)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )
      
    elif method=="sqDiff+D":
      text="(c)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arialbd.ttf'), textsize )
      
    elif method=="vgg":
      text="(d)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )
      
    elif method=="vgg+D":
      text="(e)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arialbd.ttf'), textsize )
      
    elif method=="linear_in_sample":
      text="(f)"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )
    
    method_description = Image.new(color, (int(600/1024*dims[1]),dims[1]), 'white')
    draw = ImageDraw.Draw(method_description)
    draw.text( (int(1/1024*dims[1]), int(300/1024*dims[1])), text, fill='black', font=arialFont)
    dst.paste(method_description, (0,(k_method)*dims[1]))
       
    [imgs,cost,critics] = geodesics_dict[method]

    imgs = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
    imgs = imgs.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

    for k_path in range(n_pts_on_geodesic):
	    
      if dims[0] == 1:
        img_plot = Image.fromarray( imgs[k_path,:,:,0], color )
      else: 
        img_plot = Image.fromarray( imgs[k_path], color )
			
      dst.paste(img_plot, (int(600/1024*dims[1])+dims[1]*k_path, dims[1]*k_method))


  if dims[1] > 256:
    scaling = int(dst.width/4),int(dst.height/4)
    dst = dst.resize(scaling,resample=Image.BILINEAR)	
	

  dst.save( 'results/%s/images/paths_%s.jpg' % (experiment_id, optional_run_id) ) 
  return None


##########################################################################
##########################################################################



def plot_critics(geodesics_dict, methods, experiment_id, optional_run_id):


  k = 0
  color_marker = ['rd-','k+-','bv-','g^-','yo-','ms-','k*-']
  
  for method in methods:

    [_, _,critics] = geodesics_dict[method]
    critics_plot = [item for sublist in critics for item in sublist]
    plt.plot(range(len(critics_plot)),critics_plot,color_marker[k],label=method)
    k = k + 1

  plt.ylabel('Critic value')
  plt.legend()
  plt.savefig('results/%s/images/critics_%s.jpg' % (experiment_id, optional_run_id) )
  plt.close()


  return None

##########################################################################
##########################################################################



def plot_sqDiff(geodesics_dict, methods, experiment_id, optional_run_id):


  k = 0
  color_marker = ['rd-','k+-','bv-','g^-','yo-','ms-','k*-']
  
  for method in methods:

    [_, sqDiff,_] = geodesics_dict[method]
    plt.plot(range(len(sqDiff)),sqDiff,color_marker[k],label=method)
    k = k + 1

  plt.ylabel('Square differences')
  plt.legend()
  plt.savefig('results/%s/images/square_differences_%s.jpg' % (experiment_id, optional_run_id) )
  plt.close()


  return None


##########################################################################
##########################################################################


def prepare_GAN_nets(sess, model):
    with open( model, 'rb' ) as file:
        G, D, Gs = pickle.load( file )


    # Take out the variables that correspond to the minibatch standard deviation and set them to zero
    #print([v for v in tf.global_variables()])
    if "mnist" in model:
      D44ConvLayer = [v for v in tf.global_variables() if v.name == "D/4x4/Conv/weight:0"][0]
      D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 128, :].assign( tf.zeros( (3, 3, 128) ) )

      D44ConvLayer_woMiniBatchStd = sess.run( D44ConvLayer_killMiniBatchStd)
      tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )
      
    else:
      D44ConvLayer = [v for v in tf.global_variables() if v.name == "D_paper/4x4/Conv/weight:0"][0]
      D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

      D44ConvLayer_woMiniBatchStd = sess.run( D44ConvLayer_killMiniBatchStd)
      tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )

    return G,D, Gs


##########################################################################
##########################################################################


def prepare_VGG_layers(sess):
    
    K.set_session(sess)
    
    vgg = VGG19(weights='imagenet', include_top=False)
   
    vgg_block1_conv2 = keras.Sequential(vgg.layers[:3])
    vgg_block2_conv2 = keras.Sequential(vgg.layers[:6])
    vgg_block3_conv2 = keras.Sequential(vgg.layers[:9])
    vgg_block4_conv2 = keras.Sequential(vgg.layers[:14])
    vgg_block5_conv2 = keras.Sequential(vgg.layers[:19])
    
    del vgg
    
    return vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv2, vgg_block5_conv2
    
##########################################################################
##########################################################################
# Convenience func that casts all of its arguments to tf.float32.
def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


def safe_log(x):
    return tf.log( x + 1e-8 )


def pixel_norm(x, epsilon=1e-8):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


##########################################################################
##########################################################################
    
    
def video_generation(start, end, polynomial_degree, dim_latent, model, methods, n_frames, fps, gpu_id, experiment_id, optional_run_id, global_seed, video_percentage, **configurations):
  

  print("Starting video generation\n")
  
  #################
  ###### SETUP

  ###### PARAMETERS TO SET

  batch_size=16

  ##### Derived parameters

  seed_collection = np.random.RandomState(global_seed).randn( 1000, 512 ).astype( 'float32' )

  latent_start = seed_collection[start]
  latent_end = seed_collection[end]
  
  if configurations['START_SEED_OFF']:    
      latent_start = np.float32(np.load('results/'+experiment_id+'/coefficients/_'+optional_run_id+'_'+str(start)+'.npy'))
     
  if configurations['END_SEED_OFF']:
      latent_end = np.load('results/'+experiment_id+'/coefficients/_'+optional_run_id+'_'+str(end)+'.npy')  
  
  batch_size = np.min([batch_size,n_frames])


  ########################
  ####    Start session
  
  sess = tf.InteractiveSession()

  ########################
  ####    Load generator G
    
  G, D, Gs = prepare_GAN_nets(sess, model)
  
  # For measuring squared differences in VGG layer
  K.set_session(sess)  
  vgg = VGG19(weights='imagenet', include_top=False)
  #vgg_block1_conv2 = keras.Sequential([preproLayer]+vgg.layers[:3])
  vgg_block1_conv2 = keras.Sequential(vgg.layers[:3])
  vgg_block2_conv2 = keras.Sequential(vgg.layers[:6])
  vgg_block3_conv2 = keras.Sequential(vgg.layers[:9])
  vgg_block4_conv2 = keras.Sequential(vgg.layers[:14])
  vgg_block5_conv2 = keras.Sequential(vgg.layers[:19])
    
  
 
  ########################
  ####  Set up graph for generated images
  
 
  plchldr = G.input_templates[0]
  lbls_plchldr = G.input_templates[1]
  gnrtd = G.get_output_for( plchldr, lbls_plchldr, is_training=False )
  critics = D.get_output_for( gnrtd, is_training=False )
  
  ### Extend to VGG layers
  img_data = tf.transpose(gnrtd, perm=[0,2,3,1])
        
  img_data = tf.image.resize_bilinear(img_data,(224,224))
  img_data = (img_data + 1.0) / 2.0 * 255.0 
  img_data = img_data[:,:,:,::-1]
  mean = [103.939, 116.779, 123.68]
  broadcast_shape = tf.where([True, False, False, False],
                           tf.shape(img_data), [0, 224, 224, 3])
  img_data = tf.broadcast_to(img_data, broadcast_shape)
  
  block1_conv2_features = vgg_block1_conv2(img_data)
  block2_conv2_features = vgg_block2_conv2(img_data)
  block3_conv2_features = vgg_block3_conv2(img_data)
  block4_conv2_features = vgg_block4_conv2(img_data)
  block5_conv2_features = vgg_block5_conv2(img_data)
  


  ##############################################
  ##### Collect generated images for each method
  ##############################################

  critics_dict={}
  sqDiff_dict={}
  vgg_sqDiff_dict={}
  
  
  for method in methods:
    imgs_all=np.empty((0,3,1024,1024))
    critics_all=np.empty((0,1))
    
    block1s=np.empty((0,224,224,64))
    block2s=np.empty((0,112,112,128))
    block3s=np.empty((0,56,56,256))
    block4s=np.empty((0,28,28,512))
    block5s=np.empty((0,14,14,512))
      
    if method == "linear_in_sample":
    
      print("Collecting Images along geodesics")
      print("Method: linear_in_sample", end="\r")
    
      latents_batch = np.asarray([latent_start,latent_end])
      lbls_batch = np.zeros( [2] + Gs.input_shapes[1][1:] )
    
      imgs_start_end =  tf.get_default_session().run(gnrtd, feed_dict={plchldr : latents_batch , lbls_plchldr :lbls_batch})
      theta = np.linspace( 0.0, 1.0, num=n_frames)
      imgs_all = np.asarray([(imgs_start_end[0] * (1 - theta[i]) + imgs_start_end[1]* theta[i]) for i in range( np.shape( theta )[0] )],dtype=np.float32 ) 	       
    	
      
     ##### Save images
      imgsPlot = np.clip( np.rint( (imgs_all + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )
      imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
      for idx in range( imgsPlot.shape[0] ):
        Image.fromarray( imgsPlot[idx], 'RGB' ).save( 'results/'+experiment_id+'/videos/tmp/'+method+'_'+optional_run_id+'%d.jpg' % idx )
        percent = int(float(idx)/imgsPlot.shape[0]*100)
        if idx % batch_size == 0: 
          print( "Method: "+method+"; Status: " + str( percent ) + ' %' , end="\r" )   
      
    else:
      if method == "linear":
        theta = np.linspace( 0.0, 1.0, num=n_frames)
        latents = np.asarray([(latent_start * (1 - theta[i]) + latent_end * theta[i]) for i in range( np.shape( theta )[0] )],dtype=np.float32 )

      else:
  
        ####### Load geodesic coefficients
        coefficients_file = 'results/'+experiment_id+'/coefficients/'+method+'_'+optional_run_id+'.npy'
  
        coefficients_free = np.load(coefficients_file)   # size=(polynomial_degree - 1, dim_latent) ).astype( "float32" )

        ###### For t=1/N[0,1,2,...N-1] get gamma(t)

        fac1 = [2.0 ** i - 1.0 for i in range( 2, polynomial_degree+ 1 )]
        # =[3, 7] for poly=3 
        fac1_t = np.reshape(fac1,(polynomial_degree - 1, 1))


        fac2 = [2.0 ** i - 2.0 for i in range( 2, polynomial_degree + 1 )]
        # =[2,6] for poly=3
        fac2_t = np.reshape(fac2,(polynomial_degree - 1, 1))


        c1 =     np.reshape( latent_end,   (1, dim_latent) ) - np.reshape( latent_start, (1, dim_latent) )- np.reshape( np.sum( np.multiply(fac1_t, coefficients_free), axis=0 ), (1, dim_latent) )
		
        c0 = 2 * np.reshape( latent_start, (1, dim_latent) ) - np.reshape( latent_end, (1, dim_latent) )  + np.reshape( np.sum( np.multiply(fac2_t, coefficients_free), axis=0 ), (1, dim_latent) )


        coefficients = np.concatenate( [c0, c1, coefficients_free], axis=0 )

        # Initialize parameter variable of size interpolation_degree times dimensions_noise space
        # Find interpolation points on curve dependent on the coefficients                
        
        interpolation_matrix_entries = np.zeros( shape=(n_frames, polynomial_degree + 1) )
        for i in range( n_frames ):
          for j in range( polynomial_degree + 1 ):
            interpolation_matrix_entries[i, j] = (1.0 + float( i ) / (n_frames-1)) ** j

        latents = np.dot(interpolation_matrix_entries,coefficients)

      ## Finished cases to compute latents
      ##############################################################################
      
      # Save latent vector is not entire video should be used
      if video_percentage < 100:
          np.save('results/'+experiment_id+'/coefficients/_'+optional_run_id+'_'+str(end), latents[int(n_frames*video_percentage/100)]) 
  
  
      ##############################################################################
      ##### Compute images under G
      n_batches = int(np.ceil(float(latents.shape[0])/batch_size))
  
      print("")
      for i in range(n_batches):

        percent = int(float(i+1)/n_batches * 100)
  
        print( "Method: "+method+"; Status: " + str( percent ) + ' %' , end="\r" )

        if i*batch_size < latents.shape[0]:
          end = np.min([latents.shape[0],(i+1)*batch_size])
      

        latents_batch = latents[i*batch_size : end, : ]

        lbls_batch = np.zeros([end-i*batch_size] + Gs.input_shapes[1][1:] ) 
    
        imgs_batch, critic_values_batch,block1_batch, block2_batch, block3_batch, block4_batch,block5_batch =  tf.get_default_session().run([gnrtd,critics, block1_conv2_features, block2_conv2_features,block3_conv2_features,block4_conv2_features,block5_conv2_features], feed_dict={plchldr : latents_batch , lbls_plchldr :lbls_batch})
        
        critics_all=np.append(critics_all,critic_values_batch[0],axis=0)
        imgs_all=np.append(imgs_all,imgs_batch,axis=0)
        
        
        block1s=np.append(block1s,block1_batch,axis=0)
        block2s=np.append(block2s,block2_batch,axis=0)
        block3s=np.append(block3s,block3_batch,axis=0)
        block4s=np.append(block4s,block4_batch,axis=0)
        block5s=np.append(block5s,block5_batch,axis=0)
           
  
	      ##### Save images
        imgsPlot = np.clip( np.rint( (imgs_batch + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )
        imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
        for idx in range( imgsPlot.shape[0] ):
          Image.fromarray( imgsPlot[idx], 'RGB' ).save('results/'+experiment_id+'/videos/tmp/'+method+'_'+optional_run_id+'%d.jpg' % (idx+i*batch_size) )
          
  
          
    #############################    
    ## Collect statistics
    
    ## Critics
    critics_dict[method]=critics_all
   	
    ## square differences 
    dataset_dims= (imgs_all.shape)[1:]      
    normalization = reduce(lambda x,y:x*y, dataset_dims)

    squared_differences = np.sum( ( imgs_all[1:, :, :, :] - imgs_all[:-1, :, :, :] )**2 , axis=(1,2,3))* 1.0/normalization      
    
    sqDiff_dict[method]=squared_differences
    
    ## square differences in vgg layer
    vgg_differences = 0
    vgg_differences += 1.0/(224*224*64)* np.sum((block1s[:-1]-block1s[1:])**2,axis=(1,2,3))*1e-6
    vgg_differences += 1.0/(112*112*128)* np.sum((block2s[:-1]-block2s[1:])**2,axis=(1,2,3))*1e-6
    vgg_differences += 1.0/(56*56*256)* np.sum((block3s[:-1]-block3s[1:])**2,axis=(1,2,3))*1e-6
    vgg_differences += 1.0/(28*28*512)* np.sum((block4s[:-1]-block4s[1:])**2,axis=(1,2,3))*1e-6
    vgg_differences += 1.0/(14*14*512)* np.sum((block5s[:-1]-block5s[1:])**2,axis=(1,2,3))*1e-6

    
    vgg_sqDiff_dict[method]=vgg_differences

    
  ##################
  ##Plot statistics
  plot_from_dict(critics_dict, "CriticValues", experiment_id, optional_run_id)
  plot_from_dict(sqDiff_dict, "SquareDifferences", experiment_id, optional_run_id)
  plot_from_dict(vgg_sqDiff_dict, "VGGdistance", experiment_id, optional_run_id)
           	       
  ################################
  ##### Create combined pictures and compose video
  ################################

  fontsFolder = "Font/"
  textsize=100
    
  header = Image.new('RGB', (len(methods)*1024, 160), 'white')
  draw = ImageDraw.Draw(header)

  for k_method in range(len(methods)):
      
    method = methods[k_method]
    text=""
    if method=="linear":
      text="(a) Linear"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )

    elif method=="sqDiff":
      text="(b) sqDiff"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )
      
    elif method=="sqDiff+D":
      text="(c) sqDiff+D"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arialbd.ttf'), textsize )
      
    elif method=="vgg":
      text="(d) VGG"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )
      
    elif method=="vgg+D":
      text="(e) VGG+D"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arialbd.ttf'), textsize )
      
    elif method=="linear_in_sample":
      text="(f) LinearSample"
      arialFont = ImageFont.truetype(os.path.join(fontsFolder, 'arial.ttf'), textsize )  
    
    draw.text((140+k_method*1024, 10), text, fill='black', font=arialFont)

  print("")

  dst = Image.new('RGB', (1024 * len(methods),160+1024))
  dst.paste(header, (0,1024))
    

  decrease_size_factor = 4

  height,width,layers=int((1024+160)/decrease_size_factor),int(1024*len(methods)/decrease_size_factor),3

  fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
  
  if configurations['START_SEED_OFF']:
    video_collect=cv2.VideoWriter('results/'+experiment_id+'/videos/video'+ optional_run_id +str(start)+'.avi',fourcc,fps,(width,height))
  
  else:
    video_collect=cv2.VideoWriter('results/'+experiment_id+'/videos/video'+ optional_run_id +'.avi',fourcc,fps,(width,height))

  
  for i in range(0,int(n_frames*video_percentage/100)):
    
    percent = int(float(i)/n_frames*100)
    print("Creating video; Status: "+str(percent)+' %', end='\r')

    dst = Image.new('RGB', (1024 * len(methods),160+1024))
    dst.paste(header, (0,1024))
    
    for k_method in range(len(methods)):
        method = methods[k_method]
        # imgs = []
        img_plot = Image.open('results/'+experiment_id+'/videos/tmp/'+method+'_'+optional_run_id+str(i)+'.jpg')

        dst.paste(img_plot, (1024*k_method,0))
    
    size = int(dst.width/decrease_size_factor),int(dst.height/decrease_size_factor)
    dst_small = dst.resize(size,resample=Image.BILINEAR)
    dst_small_array = np.array(dst_small)
    dst_small_array_colors = cv2.cvtColor(dst_small_array, cv2.COLOR_BGR2RGB)
    
    video_collect.write(dst_small_array_colors)
    
  video_collect.release()
  
  cv2.destroyAllWindows()

  print("")


  ####################################################################
  ## Plot critic values along video curve
  
  

  ###########################
  ##### Empty tmp folder of images
  ###########################

  files = glob.glob('results/'+experiment_id+'/videos/tmp/*')
  for f in files:
    os.remove(f) 

    
  sess.close()
  tf.reset_default_graph()
  
  print("Done\n------------------------\n")
  
  return None
  
###########################################################################
## Plotting of statistics:

def plot_from_dict(values_dict, text,experiment_id, optional_run_id):
  for key, value in values_dict.items():
    epochs = list(np.linspace(0,1,len(value)))
    plt.plot(epochs,value,label=key)
  plt.ylabel(text)
  plt.legend()
  plt.savefig('results/%s/images/%s_smooth_%s.jpg' % (experiment_id, text, optional_run_id) )
  plt.close()
  return None
  



  
###########################################################################
##########################################################################
### The following are post-processing functions 
### that can be run from a python environment
### but are not used in the main code
##########################################################################    
##########################################################################    
    
  