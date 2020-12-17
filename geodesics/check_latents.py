import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pickle
import PIL.Image


import geodesics.tfutil as tfutil 
import numpy as np
import tensorflow as tf
import geodesics.utils as utils
from geodesics.plotting import *
from geodesics.configs import *

print(methods)

#seed = 0


# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open( 'models/karras2018iclr-celebahq-1024x1024.pkl', 'rb' ) as file:
#with open( 'models/network-snapshot-005920.pkl', 'rb' ) as file:
    G, D, Gs = pickle.load( file )

D44ConvLayer = [v for v in tf.global_variables() if v.name == "D_paper/4x4/Conv/weight:0"][0]
D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

D44ConvLayer_woMiniBatchStd = tf.get_default_session().run(D44ConvLayer_killMiniBatchStd)
tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )

labels = np.zeros( [no_pts_on_geodesic] + G.input_shapes[1][1:] )
latents_plchldr = G.input_templates[0]
labels_plchldr = G.input_templates[1]
gnrtd = G.get_output_for( latents_plchldr, labels_plchldr, is_training=False ) # always (:,3,1024,1024)
fake_scores_out, fake_labels_out = utils.fp32( D.get_output_for( gnrtd, is_training=False ) )


geodesics_dict = {}

for method in methods:

    latents = np.load('models/saved_latents_' + method + '.npy')
    
    geodesics_dict[method] =  tf.get_default_session().run([gnrtd, fake_scores_out, fake_scores_out ], feed_dict={latents_plchldr : latents, labels_plchldr :labels })
    imgs, critic_values, _ = geodesics_dict[method]
    print("Critic values for " + method + ":")
    print(critic_values)
    #plot_geodesic(imgs, method + '_remake')

plot_geodesic_comparison(geodesics_dict)
plot_critics(geodesics_dict)
