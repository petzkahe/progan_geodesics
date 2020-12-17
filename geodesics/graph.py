import numpy as np
import pickle
import tensorflow as tf

from geodesics.configs import *
import geodesics.tfutil as tfutil
import geodesics.utils as utils

import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Input



def import_linear_graph(G,D):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    

    return samples, squared_differences, latents, labels, critic_values


def import_linear_in_sample_graph(G,D):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents, labels, is_training=False )

    samples_start = tf.broadcast_to(samples[0,:,:,:],[no_pts_on_geodesic,3,1024,1024])
    samples_end = tf.broadcast_to(samples[-1,:,:,:],[no_pts_on_geodesic,3,1024,1024])
    
    theta_tensor = tf.constant(np.linspace( 0.0, 1.0, num=no_pts_on_geodesic ),dtype=tf.float32,shape=[no_pts_on_geodesic,1,1,1])
    new_samples = samples_start * (1-theta_tensor) + samples_end* theta_tensor

    

    critic_values,_ = utils.fp32(D.get_output_for(new_samples, is_training=False))

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( new_samples[1:, :, :, :] - new_samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    

    return new_samples, squared_differences, latents, labels, critic_values


def import_disc_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values, _ = utils.fp32( D.get_output_for( samples, is_training=False ) )

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    

    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value_found,max_ideal_critic_value)

    small_eps = 0.01

    #### Make critic values positive by shifting by an offset, at offset the value shall be one
    # positified_critic_values = tf.clip_by_value( critic_values + offset,small_eps, np.infty)
    #positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )
    positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )


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


    # The disc loss is a weighted average of the one over the critic and the Jacobian length, so that
    # we enforce both small path length as well as real-looking images

    objective = tf.reduce_sum(tf.square(critic_objective)) # + tf.reduce_sum(squared_differences )

    return samples, squared_differences, objective, latents, labels, critic_objective, critic_values


def import_mse_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    
    objective = tf.reduce_sum(squared_differences)


    return samples, squared_differences, objective, latents, labels, critic_values




def import_vgg_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    
    img_data = tf.reshape(samples,[no_pts_on_geodesic,1024,1024,3])
    img_data = tf.image.resize_bilinear(img_data,(224,224))
    img_data = (img_data + 1.0) / 2.0 * 255.0 
    #img_data = tf.keras.applications.vgg19.preprocess_input(img_data)
    img_data = tf.keras.layers.Lambda(lambda x : tf.keras.applications.vgg19.preprocess_input(x))(img_data)

    model= VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224,3)))
    block1_conv2 = keras.Sequential(model.layers[:3])
    block2_conv2 = keras.Sequential(model.layers[:6])
    block3_conv2 = keras.Sequential(model.layers[:9])
    block4_conv4 = keras.Sequential(model.layers[:16])
    block5_conv4 = keras.Sequential(model.layers[:21])
    block1_conv2_features = block1_conv2(img_data)
    block2_conv2_features = block2_conv2(img_data)
    block3_conv2_features = block3_conv2(img_data)
    block4_conv4_features = block4_conv4(img_data)
    block5_conv4_features = block5_conv4(img_data)
    block1_conv2_length = tf.size(block1_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block2_conv2_length = tf.size(block2_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block3_conv2_length = tf.size(block3_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block4_conv4_length = tf.size(block4_conv4_features,out_type=tf.float32)/no_pts_on_geodesic
    block5_conv4_length = tf.size(block5_conv4_features,out_type=tf.float32)/no_pts_on_geodesic
    squared_differences_vgg = 0
    squared_differences_vgg += 1.0/block1_conv2_length * tf.reduce_sum(tf.square((block1_conv2_features[:-1]-block1_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block2_conv2_length * tf.reduce_sum(tf.square((block2_conv2_features[:-1]-block2_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block3_conv2_length * tf.reduce_sum(tf.square((block3_conv2_features[:-1]-block3_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block4_conv4_length * tf.reduce_sum(tf.square((block4_conv4_features[:-1]-block4_conv4_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block5_conv4_length * tf.reduce_sum(tf.square((block5_conv4_features[:-1]-block5_conv4_features[1:])),axis=[1,2,3])


    objective = tf.reduce_sum(squared_differences_vgg)


    return samples, squared_differences, objective, latents, labels, critic_values


def import_vgg_plus_disc_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]


    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    #squared_differences = tf.multiply( tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] )), utils.fp32( 1.0 / (1024 * 1024 * 3) ) )
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))
    
    # vgg part
    img_data = tf.reshape(samples,[no_pts_on_geodesic,1024,1024,3])
    img_data = tf.image.resize_bilinear(img_data,(224,224))
    img_data = (img_data + 1.0) / 2.0 * 255.0 
    img_data = tf.keras.layers.Lambda(lambda x : tf.keras.applications.vgg19.preprocess_input(x))(img_data)
    #img_data = tf.keras.applications.vgg19.preprocess_input(img_data)
    
    model= VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224,3)))
    block1_conv2 = keras.Sequential(model.layers[:3])
    block2_conv2 = keras.Sequential(model.layers[:6])
    block3_conv2 = keras.Sequential(model.layers[:9])
    block4_conv4 = keras.Sequential(model.layers[:16])
    block5_conv4 = keras.Sequential(model.layers[:21])
    block1_conv2_features = block1_conv2(img_data)
    block2_conv2_features = block2_conv2(img_data)
    block3_conv2_features = block3_conv2(img_data)
    block4_conv4_features = block4_conv4(img_data)
    block5_conv4_features = block5_conv4(img_data)
    block1_conv2_length = tf.size(block1_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block2_conv2_length = tf.size(block2_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block3_conv2_length = tf.size(block3_conv2_features,out_type=tf.float32)/no_pts_on_geodesic
    block4_conv4_length = tf.size(block4_conv4_features,out_type=tf.float32)/no_pts_on_geodesic
    block5_conv4_length = tf.size(block5_conv4_features,out_type=tf.float32)/no_pts_on_geodesic
    squared_differences_vgg = 0
    squared_differences_vgg += 1.0/block1_conv2_length * tf.reduce_sum(tf.square((block1_conv2_features[:-1]-block1_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block2_conv2_length * tf.reduce_sum(tf.square((block2_conv2_features[:-1]-block2_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block3_conv2_length * tf.reduce_sum(tf.square((block3_conv2_features[:-1]-block3_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block4_conv4_length * tf.reduce_sum(tf.square((block4_conv4_features[:-1]-block4_conv4_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1.0/block5_conv4_length * tf.reduce_sum(tf.square((block5_conv4_features[:-1]-block5_conv4_features[1:])),axis=[1,2,3])


    # disc part
    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value_found,max_ideal_critic_value)
    small_eps = 0.01
    positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )
    averaged_critic_values = tf.exp(
        tf.multiply( 0.5, tf.add( utils.safe_log( positified_critic_values[1:, :] ),
                                  utils.safe_log(
                                      positified_critic_values[:-1, :] ) ) ) )
    critic_objective = tf.divide( 1.0, averaged_critic_values )
    
    objective = tf.reduce_sum( tf.square(hyper_critic_penalty*critic_objective + tf.sqrt(squared_differences_vgg) ) )
        

    return samples, squared_differences, objective, latents, labels, critic_values




def import_mse_plus_disc_graph(G,D, latents_tensor):
    latents = G.input_templates[0]
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/(1024*1024*3)))

    critic_values, _ = utils.fp32( D.get_output_for( samples, is_training=False ) )


    # disc part here

    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value_found,max_ideal_critic_value)
    small_eps = 0.01
    positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )
    averaged_critic_values = tf.exp(
        tf.multiply( 0.5, tf.add( utils.safe_log( positified_critic_values[1:, :] ),
                                  utils.safe_log(
                                      positified_critic_values[:-1, :] ) ) ) )
    critic_objective = tf.divide( 1.0, averaged_critic_values )

    

    if use_objective_from_paper:
        objective = tf.reduce_sum( tf.square(hyper_critic_penalty*critic_objective + tf.sqrt(squared_differences) ) )
        
    else:
        objective = hyper_critic_penalty* tf.reduce_sum(critic_objective) + tf.reduce_sum(squared_differences )
    
    

    return samples, squared_differences, objective, latents, labels, critic_objective, critic_values



def parameterize_line( latent_start, latent_end ):

    theta = np.linspace( 0.0, 1.0, num=no_pts_on_geodesic )
    latents = np.asarray([(latent_start * (1 - theta[i]) + latent_end * theta[i]) for i in range( np.shape( theta )[0] )],dtype=np.float32 )
    #latents = latents * tf.rsqrt(tf.reduce_sum(tf.square(latents), axis=1, keepdims=True) + 1e-8)
    #latents = latents * np.rsqrt()
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




