import numpy as np
import pickle
import tensorflow as tf
from functools import reduce

import utils

import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Input



def import_linear_graph(G,D):
    
    

    latents = G.input_templates[0]
    
    labels = G.input_templates[1]
    
    samples = G.get_output_for( latents, labels, is_training=False )
    
    dataset_dims= samples.get_shape().as_list()[1:]
    
    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    normalization = reduce(lambda x,y:x*y, dataset_dims)
    
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    

    return samples, squared_differences, latents, labels, critic_values


def import_linear_in_sample_graph(G,D, n_pts_on_geodesic, **configurations):
    latents = G.input_templates[0]
    labels = G.input_templates[1]

    samples = G.get_output_for( latents, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]
    
    samples_start = tf.broadcast_to(samples[0,:,:,:],[n_pts_on_geodesic,*dataset_dims])
    samples_end = tf.broadcast_to(samples[-1,:,:,:],[n_pts_on_geodesic,*dataset_dims])
    
    theta_tensor = tf.constant(np.linspace( 0.0, 1.0, num=n_pts_on_geodesic ),dtype=tf.float32,shape=[n_pts_on_geodesic,1,1,1])
    new_samples = samples_start * (1-theta_tensor) + samples_end* theta_tensor

    
    critic_values,_ = utils.fp32(D.get_output_for(new_samples, is_training=False))
    
    normalization = reduce(lambda x,y:x*y, dataset_dims)
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( new_samples[1:, :, :, :] - new_samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    

    return new_samples, squared_differences, latents, labels, critic_values


def import_disc_graph(G,D, latents_tensor, min_critic_value, max_ideal_critic_value, **configurations):
    
    labels = G.input_templates[1]
    
    samples = G.get_output_for( latents_tensor, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]
    
    critic_values, _ = utils.fp32( D.get_output_for( samples, is_training=False ) )

    normalization = reduce(lambda x,y:x*y, dataset_dims)

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    

    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value,max_ideal_critic_value)

    offset =  min_critic_value - 10  
    scaling = 1./(max_ideal_critic_value - offset)

    small_eps = 0.01

    #### Make critic values positive by shifting by an offset, at offset the value shall be one
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
    critic_objective = tf.divide( 1.0, averaged_critic_values )


    # The disc loss is a weighted average of the one over the critic and the Jacobian length, so that
    # we enforce both small path length as well as real-looking images

    objective = tf.reduce_sum(tf.square(critic_objective)) # + tf.reduce_sum(squared_differences )

    return samples, squared_differences, objective, labels, critic_objective, critic_values


def import_sqDiff_graph(G,D,latents_tensor):
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]

    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))
    
    normalization = reduce(lambda x,y:x*y, dataset_dims)
    
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    
    objective = tf.reduce_sum(squared_differences)


    return samples, squared_differences, objective, labels, critic_values




def import_vgg_graph(G,D, latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv2, vgg_block5_conv2, n_pts_on_geodesic,**configurations ):
    
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]
    
    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    normalization = reduce(lambda x,y:x*y, dataset_dims)

    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    
     

    img_data = tf.transpose(samples, perm=[0,2,3,1])
        
    img_data = tf.image.resize_bilinear(img_data,(224,224))
    img_data = (img_data + 1.0) / 2.0 * 255.0 
    img_data = img_data[:,:,:,::-1]
    mean = [103.939, 116.779, 123.68]
    img_data = img_data - tf.broadcast_to(mean,shape=[n_pts_on_geodesic,224,224,3])

    block1_conv2_features = vgg_block1_conv2(img_data)
    block2_conv2_features = vgg_block2_conv2(img_data)
    block3_conv2_features = vgg_block3_conv2(img_data)
    block4_conv2_features = vgg_block4_conv2(img_data)
    block5_conv2_features = vgg_block5_conv2(img_data)
    block1_conv2_length = tf.size(block1_conv2_features,out_type=tf.float32)/n_pts_on_geodesic 
    block2_conv2_length = tf.size(block2_conv2_features,out_type=tf.float32)/n_pts_on_geodesic
    block3_conv2_length = tf.size(block3_conv2_features,out_type=tf.float32)/n_pts_on_geodesic
    block4_conv2_length = tf.size(block4_conv2_features,out_type=tf.float32)/n_pts_on_geodesic
    block5_conv2_length = tf.size(block5_conv2_features,out_type=tf.float32)/n_pts_on_geodesic
    squared_differences_vgg = 0
    squared_differences_vgg += 1e-6/block1_conv2_length * tf.reduce_sum(tf.square((block1_conv2_features[:-1]-block1_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block2_conv2_length * tf.reduce_sum(tf.square((block2_conv2_features[:-1]-block2_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block3_conv2_length * tf.reduce_sum(tf.square((block3_conv2_features[:-1]-block3_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block4_conv2_length * tf.reduce_sum(tf.square((block4_conv2_features[:-1]-block4_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block5_conv2_length * tf.reduce_sum(tf.square((block5_conv2_features[:-1]-block5_conv2_features[1:])),axis=[1,2,3])


    objective = tf.reduce_sum(squared_differences_vgg)


    return samples, squared_differences, objective, labels, critic_values


def import_vgg_plus_D_graph(G,D, latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv2, vgg_block5_conv2,n_pts_on_geodesic, min_critic_value,max_ideal_critic_value, hyper_vgg, **configurations):
    
    labels = G.input_templates[1]

    samples = G.get_output_for( latents_tensor, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]
    
    critic_values,_ = utils.fp32(D.get_output_for(samples, is_training=False))

    normalization = reduce(lambda x,y:x*y, dataset_dims)
    
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))
    
    
    # vgg part
    img_data = tf.transpose(samples, perm=[0,2,3,1])
    img_data = tf.image.resize_bilinear(img_data,(224,224))
    img_data = (img_data + 1.0) / 2.0 * 255.0 
    check = preprocess_input(img_data)
    img_data = img_data[:,:,:,::-1]
    mean = [103.939, 116.779, 123.68]
    img_data = img_data - tf.broadcast_to(mean,shape=[n_pts_on_geodesic,224,224,3])
        
    block1_conv2_features = vgg_block1_conv2(img_data)
    block2_conv2_features = vgg_block2_conv2(img_data)
    block3_conv2_features = vgg_block3_conv2(img_data)
    block4_conv2_features = vgg_block4_conv2(img_data)
    block5_conv2_features = vgg_block5_conv2(img_data)
    block1_conv2_length = tf.size(block1_conv2_features,out_type=tf.float32)
    block2_conv2_length = tf.size(block2_conv2_features,out_type=tf.float32)
    block3_conv2_length = tf.size(block3_conv2_features,out_type=tf.float32)
    block4_conv2_length = tf.size(block4_conv2_features,out_type=tf.float32)
    block5_conv2_length = tf.size(block5_conv2_features,out_type=tf.float32)

    squared_differences_vgg = 0
    squared_differences_vgg += 1e-6/block1_conv2_length * tf.reduce_sum(tf.square((block1_conv2_features[:-1]-block1_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block2_conv2_length * tf.reduce_sum(tf.square((block2_conv2_features[:-1]-block2_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block3_conv2_length * tf.reduce_sum(tf.square((block3_conv2_features[:-1]-block3_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block4_conv2_length * tf.reduce_sum(tf.square((block4_conv2_features[:-1]-block4_conv2_features[1:])),axis=[1,2,3])
    squared_differences_vgg += 1e-6/block5_conv2_length * tf.reduce_sum(tf.square((block5_conv2_features[:-1]-block5_conv2_features[1:])),axis=[1,2,3])


    # disc part
    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value,max_ideal_critic_value)
    
    offset =  min_critic_value - 10  
    scaling = 1./(max_ideal_critic_value - offset)
    
    small_eps = 0.01
        
    positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )
    averaged_critic_values = tf.exp(
        tf.multiply( 0.5, tf.add( utils.safe_log( positified_critic_values[1:, :] ),
                                  utils.safe_log(
                                      positified_critic_values[:-1, :] ) ) ) )
    critic_objective = tf.divide( 1.0, averaged_critic_values )
    
    objective = tf.reduce_sum( tf.square(hyper_vgg*critic_objective + tf.sqrt(squared_differences_vgg) ) )
        
    return samples, squared_differences, objective, labels, critic_values




def import_sqDiff_plus_D_graph(G,D, latents_tensor, min_critic_value, max_ideal_critic_value, hyper_sqDiff, **configurations):

    labels = G.input_templates[1]
    
    samples = G.get_output_for( latents_tensor, labels, is_training=False )
    dataset_dims= samples.get_shape().as_list()[1:]
    

    normalization = reduce(lambda x,y:x*y, dataset_dims)
    
    squared_differences = tf.multiply(tf.reduce_sum( tf.square( samples[1:, :, :, :] - samples[:-1, :, :, :] ) , axis=[1,2,3]), utils.fp32(1.0/normalization))

    critic_values, _ = utils.fp32( D.get_output_for( samples, is_training=False ) )


    # disc part here

    critic_values_capped = tf.clip_by_value(critic_values,min_critic_value,max_ideal_critic_value)
    
    offset =  min_critic_value - 10  
    scaling = 1./(max_ideal_critic_value - offset)
    
    small_eps = 0.01
    positified_critic_values = tf.clip_by_value(scaling * (critic_values_capped - offset), small_eps, 1-small_eps )
    averaged_critic_values = tf.exp(
        tf.multiply( 0.5, tf.add( utils.safe_log( positified_critic_values[1:, :] ),
                                  utils.safe_log(
                                      positified_critic_values[:-1, :] ) ) ) )
    critic_objective = tf.divide( 1.0, averaged_critic_values )

    


    objective = tf.reduce_sum( tf.square(hyper_sqDiff*critic_objective + tf.sqrt(squared_differences) ) )
            
    return samples, squared_differences, objective, labels, critic_objective, critic_values


def adjust_for_spherical(start,end,**configurations):

  # adjust start and end to have norm equal to sqrt of dim latent
  # find sampling points on curve so that, for a line, the points are equidistant on a sqrt(dim latent) sphere
  # return start, end, and sampling pattern in [0,1]

  equidistant_points = np.linspace( 0, 1.0, num=n_pts_on_geodesic )

  if SPHERICAL_INTERPOLATION_ON == True:
  	start_unit = start/np.linalg.norm(start)
  	end_unit = end/np.linalg.norm(start)
	angle = np.clip(np.arccos( np.clip( np.dot(start_unit,end_unit),-1.0,1.0) ),10 ** -3,np.pi - 10 ** -3)
	sampling_points_on_curve = 0.5 + 0.5*np.arctan(angle*(equidistant_points-0.5)) / np.arctan(angle/2)
	sampling_points_on_curve[0] = 0
	sampling_points_on_curve[-1] = 1
  	start = start_unit*np.sqrt(dim_latent)
  	end = end_unit*np.sqrt(dim_latent)
  else:
    sampling_points_on_curve = equidistant_points
     
  return start, end, sampling_points_on_curve


def parameterize_line( latent_start, latent_end, n_pts_on_geodesic , **configurations):

    #theta = np.linspace( 0, 1.0, num=n_pts_on_geodesic )
    latent_start, latent_end, theta = adjust_for_spherical(latent_start, latent_end, **configurations)

    latents = np.asarray([(latent_start * (1 - theta[i]) + latent_end * theta[i]) for i in range( np.shape( theta )[0] )],dtype=np.float32 )
    return latents




def parameterize_curve( latent_start, latent_end, n_pts_on_geodesic, coefficient_init, polynomial_degree, dim_latent, **configurations ):
    # want to paramtererize the curve from 1 to 2, as numbers close to zero are problematic
    # we learn all 2nd degree coefficients and higher and derive linear and constant coefficientf from start and end points
    # if a(1)=c0+c1+c2=start and a(2)=c0+2c1+4c2=end, then
    # c1 = end-start- 3c2
    # c0 = 2*start - end + 2*c2

    latent_start, latent_end, theta = adjust_for_spherical(latent_start, latent_end, **configurations)

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

    interpolation_matrix_entries = np.zeros( shape=(n_pts_on_geodesic, polynomial_degree + 1) )
    for i in range( n_pts_on_geodesic ):
        for j in range( polynomial_degree + 1 ):
            #interpolation_matrix_entries[i, j] = (1.0 + float( i ) / (n_pts_on_geodesic-1)) ** j
            interpolation_matrix_entries[i, j] = (1.0 + theta[i]) ** j
    interpolation_matrix = tf.constant( interpolation_matrix_entries,
                                        shape=(n_pts_on_geodesic, polynomial_degree + 1),
                                        dtype='float32' )

    latents = tf.matmul(interpolation_matrix,coefficients)

    return latents, coefficients_free




