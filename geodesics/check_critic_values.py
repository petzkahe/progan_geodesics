import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pickle
import PIL.Image

import geodesics.tfutil as tfutil
import numpy as np
import tensorflow as tf
import geodesics.utils as utils

batch_size=32
seed = 0

max_values = []
min_values = []

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open( 'models/karras2018iclr-celebahq-1024x1024.pkl', 'rb' ) as file:
#with open( 'models/network-snapshot-005920.pkl', 'rb' ) as file:
    G, D, Gs = pickle.load( file )

# Generate latent vectors.
labels = np.zeros( [batch_size] + Gs.input_shapes[1][1:] )

latents_plchldr = G.input_templates[0]
labels_plchldr = G.input_templates[1]

# D44ConvLayer = [v for v in tf.global_variables() if v.name == "D/4x4/Conv/weight:0"][0]
# D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

# D44ConvLayer_woMiniBatchStd = tf.get_default_session().run(D44ConvLayer_killMiniBatchStd)
# tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )


# Generate latent vectors.
#latents = np.random.RandomState( 400 ).randn( 1000, *Gs.input_shapes[0][1:] )  # 1000 random latents

#latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

for seed in range(0,100):

    print(seed)
    latents = np.random.RandomState( seed ).randn( 100, *Gs.input_shapes[0][1:] )  # 1000 random latents
    latents = latents[0:batch_size,:]  # hand-picked top-10


    gnrtd = G.get_output_for( latents_plchldr, labels_plchldr, is_training=False ) # always (:,3,1024,1024)
    fake_scores_out, fake_labels_out = utils.fp32( D.get_output_for( gnrtd, is_training=False ) )


    [critic_values]=  tf.get_default_session().run([fake_scores_out], feed_dict={latents_plchldr : latents, labels_plchldr :labels })

    max= np.max(critic_values)
    min= np.min(critic_values)

    max_values.append(max)
    min_values.append(min)

print("Maximal critic value found is  "+ str(np.max(max_values)))
print("Minimal critic value found is  "+ str(np.min(min_values)))


