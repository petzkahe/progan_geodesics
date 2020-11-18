import pickle
import PIL.Image

import geodesics.tfutil as tfutil
import numpy as np
import tensorflow as tf
import geodesics.utils as utils

mode='random'
mode = 'linear path'


# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open( 'models/network-snapshot-005920.pkl', 'rb' ) as file:
    G, D, Gs = pickle.load( file )

# Generate latent vectors.
latents = np.random.RandomState( 400 ).randn( 1000, *Gs.input_shapes[0][1:] )  # 1000 random latents

if mode =='random': # this generates 10 random pictures
    latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10
    #latents = latents[[476, 55, 82, 886, 582, 390, 85, 339, 340, 414]]  # hand-picked top-10

elif mode == 'linear path': # this implements a path
    latents = latents[[79, 886]] # two random samples
#    print( np.linalg.norm( latents[1] - latents[0] ) )
    theta = np.linspace( 0.0, 1.0, num=16 )
    newlatents = [(latents[0] * (1 - theta[i]) + latents[1] * theta[i]) for i in range( np.shape( theta )[0] )]
    latents = np.asarray( newlatents, dtype=np.float32 )

else:
    raise NameError('Mode unknown')


lbls = np.zeros( [latents.shape[0]] + Gs.input_shapes[1][1:] )

plchldr = G.input_templates[0]
lbls_plchldr = G.input_templates[1]



curve=plchldr
# To also test how to get gradients
coefficients = tf.Variable( np.ones( 4, dtype=np.float32 ), name="coeffficient" )
curve = plchldr * tf.reduce_sum( coefficients )


gnrtd = G.get_output_for( curve, lbls_plchldr, is_training=False ) # always (:,3,1024,1024)
fake_scores_out, fake_labels_out = utils.fp32( D.get_output_for( gnrtd, is_training=False ) )


# To also
grdnt = tf.gradients( [fake_scores_out], [coefficients],
                      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N )
init = tf.initialize_variables( [coefficients] )
tf.get_default_session().run( init )


# Get fake_scores, images (and gradients)
#rslt, imgs =  tf.get_default_session().run([fake_scores_out, gnrtd], feed_dict={plchldr : latents, lbls_plchldr :lbls })
rslt, imgs,grds =  tf.get_default_session().run([fake_scores_out, gnrtd,grdnt], feed_dict={plchldr : latents, lbls_plchldr :lbls })


imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
for idx in range( imgsPlot.shape[0] ):
    PIL.Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/path%d.png' % idx )


print(D.list_layers())
D.print_layers()

print('Fake scores are '+str(rslt[:,0]))

# This works, but discriminator values depend on number of samples passed through generator, not only the esmaples itself


# Take out the variables that correspond to the minibatch standard deviation and set them to zero
D44ConvLayer = [v for v in tf.global_variables() if v.name == "D/4x4/Conv/weight:0"][0]
D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

D44ConvLayer_woMiniBatchStd = tf.get_default_session().run(D44ConvLayer_killMiniBatchStd)
tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )

#gradient = tf.get_default_session().run( grdnt, feed_dict={plchldr: latents, lbls_plchldr: lbls} )
#print( np.shape( gradient ) )

disc_values = D.run( imgs )[0]
print('Fake scores are now '+ str(disc_values[:,0]) )

rslt, imgs,grds =  tf.get_default_session().run([fake_scores_out, gnrtd,grdnt], feed_dict={plchldr : latents, lbls_plchldr :lbls })
print('Fake scores are for other method collecting gradients:  '+ str(disc_values[:,0]) )
print("Gradients are given by: "+ str(grds[0]))
