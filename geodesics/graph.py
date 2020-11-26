import numpy as np
import pickle
import tensorflow as tf

from geodesics.configs import *
import geodesics.tfutil as tfutil
import geodesics.utils as utils

CUDA_DIVISIBLE_DEVICES = 0

def import_linear_graph(G,D):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    




    return samples, squared_differences, latents, labels


def import_Jacobian_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    
    objective_Jacobian = tf.reduce_sum(squared_differences)


    return samples, squared_differences, objective_Jacobian, latents, labels




def import_proposed_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values, _ = utils.fp32( D.get_output_for( samples, is_training=False ) )

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    

    small_eps = 0.01

    #### Make critic values positive by shifting by an offset, at offset the value shall be one
    # positified_critic_values = tf.clip_by_value( critic_values + offset,small_eps, np.infty)
    positified_critic_values = tf.clip_by_value(scaling * (critic_values + offset), small_eps, 1-small_eps )

    ### For the length of each interval take combination of critic values at both ends
    # Why take average? Avoids single points with low disciminator value along the trajectory since those hurt twice
    # Why geometric mean instead of arithmetic mean? Not to cancel out very small values in mean
    averaged_critic_values = tf.exp(
        tf.multiply( 0.5, tf.add( utils.safe_log( positified_critic_values[1:, :] ),
                                  utils.safe_log(
                                      positified_critic_values[:-1, :] ) ) ) )
    #averaged_critic_values = tf.multiply( 0.5, tf.add( positified_critic_values[1:, :], positified_critic_values[:-1, :] ))

    # Part of the loss is 1/C(x)^2 with C the critic of the GAN
    #critic_denominator = tf.square(tf.clip_by_value( tf.add( averaged_critic_values, small_eps ), small_eps, np.infty ))
    critic_objective = tf.divide( 1.0, averaged_critic_values )


    # The proposed loss is a weighted average of the one over the critic and the Jacobian length, so that
    # we enforce both small path length as well as real-looking images

    objective_proposed = hyper_critic_penalty* tf.reduce_sum(critic_objective) + tf.reduce_sum(squared_differences )

    return samples, squared_differences, objective_proposed, latents, labels, critic_objective





def parameterize_line( latent_start, latent_end ):

    theta = np.linspace( 0.0, 1.0, num=no_pts_on_geodesic )
    latents = np.asarray([(latent_start * (1 - theta[i]) + latent_end * theta[i]) for i in range( np.shape( theta )[0] )],dtype=np.float32 )
    #latents = latents * tf.rsqrt(tf.reduce_sum(tf.square(latents), axis=1, keepdims=True) + 1e-8)
    return latents




def parameterize_curve( latent_start, latent_end ):
    # want to paramtererize the curve from 1 to 2, as numbers close to zero are problematic
    # we learn all 2nd degree coefficients and higher and derive linear and constant coefficientf from start and end points
    # if a(1)=c0+c1+c2=start and a(2)=c0+2c1+4c2=end, then
    # c1 = end-start- 3c2
    # c0 = 2*start - end + 2*c2


    coefficients_initializations = np.random.uniform( -coefficient_init,
                                                      coefficient_init, size=(
            polynomial_degree - 1, dim_latent) ).astype( "float32" )

    with tf.variable_scope( "Geodesics" ):
        coefficients_free = tf.Variable( initial_value=coefficients_initializations, name='coefficients', dtype=tf.float32 )

    # if degree= 2, then just get number 3
    fac1 = [2.0 ** i - 1.0 for i in range( 2, polynomial_degree+ 1 )]
    fac1_tensor = tf.reshape( tf.constant( fac1, dtype=tf.float32 ),
                                 shape=(polynomial_degree - 1, 1) )


    # if degree = 2, then just get number 2
    fac2 = [2.0 ** i - 2.0 for i in range( 2, polynomial_degree + 1 )]
    fac2_tensor = tf.reshape( tf.constant( fac2, dtype=tf.float32 ),
                                 shape=(polynomial_degree- 1, 1) )

    c1 = tf.reshape( latent_end, shape=(1, dim_latent) ) - tf.reshape( latent_start, shape=(
        1, dim_latent) ) - tf.reshape(
        tf.reduce_sum( tf.multiply( fac1_tensor, coefficients_free ), axis=0 ),
        shape=(1, dim_latent) )

    c0 = 2 * tf.reshape( latent_start, shape=(1, dim_latent) ) - tf.reshape( latent_end, shape=(
        1, dim_latent) ) + tf.reshape(
        tf.reduce_sum( tf.multiply( fac2_tensor, coefficients_free ), axis=0 ),
        shape=(1, dim_latent) )

    coefficients = tf.concat( [c0, c1, coefficients_free], axis=0 )

    # Initialize parameter variable of size interpolation_degree times dimensions_noise space
    # Find interpolation points on curve dependent on the coefficients

    interpolation_matrix_entries = np.zeros( shape=(no_pts_on_geodesic, polynomial_degree + 1) )
    for i in range( no_pts_on_geodesic ):
        for j in range( polynomial_degree + 1 ):
            interpolation_matrix_entries[i, j] = (1.0 + float( i ) / (no_pts_on_geodesic-1)) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(no_pts_on_geodesic, polynomial_degree + 1),
                                        dtype='float32' )

    latents = tf.matmul(interpolation_matrix,coefficients)
    #latents = utils.pixel_norm(latents)
    #latents = latents * tf.rsqrt(tf.reduce_sum(tf.square(latents), axis=1, keepdims=True) + 1e-8)
    return latents, coefficients_free




