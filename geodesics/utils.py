import tensorflow as tf
from geodesics.configs import *
import numpy as np

# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]



def initialize_endpoints_of_curve(initialization_mode):
    if initialization_mode == "random":

        z_start_value = np.random.uniform( low=latent_min, high=latent_max,
                                           size=[512] ).astype(
            'float32' )
        z_end_value = np.random.uniform( low=latent_min, high=latent_max,
                                         size=[512] ).astype('float32')
    else:
        raise Exception( "Initialization_mode {} not known".format( initialization_mode ) )

    return z_start_value, z_end_value

def safe_log(x):
    return tf.log( x + 1e-8 )


# def pixel_norm(x, epsilon=1e-8):
# #    with tf.variable_scope('PixelNorm'):
#     #print(tf.shape(x))
#     return x * tf.rsqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True) + epsilon)
