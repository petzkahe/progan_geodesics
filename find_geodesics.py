import tensorflow as tf
import numpy as np

import graph
from utils import *


##########################################################################
##########################################################################

def find_geodesics(global_seed, start, end, methods, experiment_id, optional_run_id, adam_beta1, adam_beta2, learning_rate, n_training_steps, n_pts_on_geodesic,model, **configurations):

    ###################################################################################################
    ## Setup
    ###################################################################################################
    
    seed_collection = np.random.RandomState(global_seed).randn( 1000, configurations['dim_latent'] ).astype( 'float32' )
    np.random.seed(global_seed)
    latent_start = seed_collection[start]
    latent_end = seed_collection[end]
    
    geodesics_dict = {}
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


    ###################################################################################################
    ## Additional optional setup 
    if configurations['START_SEED_OFF']:    
      latent_start = np.float32(np.load('results/'+experiment_id+'/coefficients/_'+optional_run_id+'_'+str(start)+'.npy'))
    
    if configurations['END_SEED_OFF']:
      latent_end = np.load('results/'+experiment_id+'/coefficients/_'+optional_run_id+'_'+str(end)+'.npy')    


    if configurations['PROJECT_ENDPOINTS_ONTO_SPHERE']:
      latent_start = latent_start/np.linalg.norm(latent_start)*np.sqrt(dim_latent)
      latent_start = latent_end/np.linalg.norm(latent_end)*np.sqrt(dim_latent)
      
    #latent_start=np.clip(latent_start,0,np.infty)
    #latent_end=np.clip(latent_end,0,np.infty)
    #latent_start_new = latent_start * (2.5) - 1.5*latent_end        
    
    #np.save('results/'+configurations["experiment_id"]+'/coefficients/_'+configurations["optional_run_id"]+'_'+str(start), latent_start_new)
    #np.save('results/'+configurations["experiment_id"]+'/coefficients/_'+configurations["optional_run_id"]+'_'+str(end), latent_end)
        
    ###################################################################################################
    
    print("\nRunning \n Experiment id: " + experiment_id+'\n Optional run_id: '+ optional_run_id)
    print("")
    
    ###################################################################################################
    ## Start finding shortest paths per method
    ###################################################################################################
    
    for method in methods:
      print("Optimizing path for " + method + "...")    
      with tf.Session(config=session_config) as sess:

        G, D, Gs = prepare_GAN_nets( sess, model )

        ################################################################################################################################################

        if method=="linear":
          latents = graph.parameterize_line(latent_start, latent_end, n_pts_on_geodesic, **configurations) # is of size (no_pts, dimension)
          images, squared_differences, latent_plchldr, labels_plchldr, critic_values = graph.import_linear_graph(G,D)
                           
        elif method=="linear_in_sample":
          latents = graph.parameterize_line(latent_start, latent_end, n_pts_on_geodesic,**configurations) # is of size (no_pts, dimension)
          images, squared_differences, latent_plchldr, labels_plchldr, critic_values = graph.import_linear_in_sample_graph(G,D, n_pts_on_geodesic, **configurations)
      
        ################################################################################################################################################
        
        else: # Training is required
      
          print("Initializing graph...")                  
        
          latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end, n_pts_on_geodesic, **configurations)
          # coefficients_free are the variables to learn, which are coefficients of the interpolating polynomial
          # latent_tensor contains the latent points on the curve
                                        
          if method == "disc":
            images, squared_differences, objective, labels_plchldr, critic_objective, critic_values = graph.import_disc_graph( G, D , latents_tensor, **configurations)

          elif method == "sqDiff":
            images, squared_differences, objective, labels_plchldr, critic_values = graph.import_sqDiff_graph( G, D , latents_tensor)
          
          elif method == "sqDiff+D":
            images, squared_differences, objective, labels_plchldr, critic_objective, critic_values = graph.import_sqDiff_plus_D_graph( G, D , latents_tensor,**configurations)
          
          elif method == "vgg":
            vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4 = prepare_VGG_layers(sess)
            images, squared_differences, objective, labels_plchldr, critic_values = graph.import_vgg_graph( G, D, latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4, n_pts_on_geodesic, **configurations)
    
          elif method == "vgg+D":
            vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4 = prepare_VGG_layers(sess)
            images, squared_differences, objective, labels_plchldr, critic_values  = graph.import_vgg_plus_D_graph( G, D , latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4, n_pts_on_geodesic, **configurations)
            
          else:
            raise Exception("Method" + method +" does not exist")  
             
          ################################################################################################################################################
          ## Begin Training
          sess.run(tf.variables_initializer([coefficients_free]))

          with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free)


          adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
          sess.run( tf.variables_initializer( adam_training_variables ) )
                        
          print("Training...")
          lbls = np.zeros( [n_pts_on_geodesic] + G.input_shapes[1][1:] )
          for iteration in range( n_training_steps ):
            _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

            if iteration % 1 == 0:
              print( "Status: " + str( int( iteration /n_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x), end="\r" )
            
          coefficients = sess.run(coefficients_free)
          np.save('results/'+experiment_id+'/coefficients/'+method+'_'+optional_run_id, coefficients)

          # finished training
          ################################################################################################################################################
      
        ################################################################################################################################################        
        # Collect results      
        lbls = np.zeros( [n_pts_on_geodesic] + G.input_shapes[1][1:] )
        if method == "linear" or method == "linear_in_sample":
          [imgs, sq_diff, critics] = sess.run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})
        else:
          [imgs, sq_diff, critics] = sess.run([images, squared_differences, critic_values],feed_dict={labels_plchldr: lbls})

        
        geodesics_dict[method]= [imgs, sq_diff, critics]
        
        print("\n... Done!\n")
      
      ################################################################################################################################################
      # Close sessions and reset graphs
      
      sess.close()

      if "vgg" in method:
        del vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, 
        tf.keras.backend.clear_session()
      tf.reset_default_graph()
         
    return geodesics_dict





