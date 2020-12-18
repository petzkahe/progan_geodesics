import tensorflow as tf
import pickle
import numpy as np
import geodesics.tfutil as tfutil
import geodesics.graph as graph
from geodesics.configs import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow.keras as keras
from tensorflow.keras.layers import Input


##########################################################################
##########################################################################


def find_geodesics(latent_start, latent_end, methods):
    geodesics_dict = {}

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    
    for method in methods:
        with tf.Session(config=session_config) as sess:

            print("Run: " + args.subfolder_path + args.file_name)
            print("Loading GAN networks")
            G, D = prepare_GAN_nets( sess )
            
            if method=="vgg":
                vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4 = prepare_VGG_layers(sess)
            
            print("Optimizing path for " + method + "...")
            
            if "vgg" in method: 
                geodesics_dict[method] = learn_geodesic_vgg(method, latent_start, latent_end, sess, G, D, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4)
            else:
                geodesics_dict[method] = learn_geodesic(method, latent_start, latent_end, sess, G, D)
            

            print("... Done!")


        tf.keras.backend.clear_session()
        sess.close()
        tf.reset_default_graph()




    return geodesics_dict




##########################################################################
##########################################################################


def learn_geodesic(method, latent_start, latent_end, sess, G, D):




        ################################################################################################################################################

        if method=="linear":

            images, squared_differences, latent_plchldr, labels_plchldr, critic_values = graph.import_linear_graph(G,D)
            latents = graph.parameterize_line(latent_start, latent_end) # is of size (no_pts, dimension)

            # identical below
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})
        ################################################################################################################################################

        elif method=="linear_in_sample":

            # A disadvantage of loading the graph by a method could be that it does not import lazily but goes through all lines of the method
            images, squared_differences, latent_plchldr, labels_plchldr, critic_values = graph.import_linear_in_sample_graph(G,D)
            
            
            latents = graph.parameterize_line(latent_start, latent_end) # is of size (no_pts, dimension)

            # identical below
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})

        ################################################################################################################################################

        elif method == "disc":

            print("Initializing graph...")
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)
            images, squared_differences, objective, latent_plchldr, labels_plchldr, critic_objective, critic_values = graph.import_disc_graph( G, D , latents_tensor)

            # identical below?
            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free
                )
            
            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )

            print("Training...")
            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

                #print(x)
                if iteration % 1 == 0:
                    print( "Status: " + str( int( iteration /geodesic_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x)) #, end="\r" )


            latents = sess.run(latents_tensor)

            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})

        ################################################################################################################################################

        elif method == "mse":

            print("Initializing graph...")
            # coefficients_free are the variables to learn, which are coefficients of the interpolating polynomial
            # latent_tensor contains the latent points on the curve
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)

            images, squared_differences, objective, latent_plchldr, labels_plchldr, critic_values = graph.import_mse_graph( G, D , latents_tensor)

            # identical below
            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free
                )


            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )


            
            
            print("Training...")
            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

                #print(x)
                if iteration % 1 == 0:
                    print( "Status: " + str( int( iteration /geodesic_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x)) #, end="\r" )

            latents = sess.run(latents_tensor)
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})

            ################################################################################################################################################

        elif method == "mse_plus_disc":

            print("Initializing graph...")
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)
            images, squared_differences, objective, latent_plchldr, labels_plchldr, critic_objective, critic_values = graph.import_mse_plus_disc_graph( G, D , latents_tensor)

            # identical below
            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free
                )

            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )


            print("Training...")
            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

                #print(x)
                if iteration % 1 == 0:
                    print( "Status: " + str( int( iteration /geodesic_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x)) #, end="\r" )


            
            latents = sess.run(latents_tensor)
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})




        else:
             raise Exception("Method" + method +" does not exist")

        
        np.save('models/' + args.subfolder_path + args.file_name + '_saved_latents_for_' + method,latents) 

        print(sq_diff)
        
        return imgs, sq_diff, critics


################################################################################################################################################
################################################################################################################################################



def learn_geodesic_vgg(method, latent_start, latent_end, sess, G, D, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4):

        if method == "vgg":

            print("Initializing graph...")
            # coefficients_free are the variables to learn, which are coefficients of the interpolating polynomial
            # latent_tensor contains the latent points on the curve
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)

            images, squared_differences, objective, latent_plchldr, labels_plchldr, critic_values = graph.import_vgg_graph( G, D , latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4)


            # identical below
            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free
                )


            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )


            print("Training...")
            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

                #print(x)
                if iteration % 1 == 0:
                    print( "Status: " + str( int( iteration /geodesic_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x)) #, end="\r" )

            latents = sess.run(latents_tensor)
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})


################################################################################################################################################

        elif method == "vgg_plus_disc":

            print("Initializing graph...")
            # coefficients_free are the variables to learn, which are coefficients of the interpolating polynomial
            # latent_tensor contains the latent points on the curve
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)

            images, squared_differences, objective, latent_plchldr, labels_plchldr, critic_values = graph.import_vgg_plus_disc_graph( G, D , latents_tensor, vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4)


            # identical below
            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("geodesic_training"):
                    trainer = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective,
                    var_list=coefficients_free
                )


            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='geodesic_training')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )


            
            print("Training...")
            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [trainer, objective], feed_dict={labels_plchldr: lbls} )

                #print(x)
                if iteration % 1 == 0:
                    print( "Status: " + str( int( iteration /geodesic_training_steps * 1000.0 )/10 ) + ' %, objective: ' + str(x)) #, end="\r" )

            latents = sess.run(latents_tensor)
            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )
            [imgs, sq_diff, critics] = tf.get_default_session().run([images, squared_differences, critic_values],feed_dict={latent_plchldr: latents, labels_plchldr: lbls})

        else:
             raise Exception("Method for vgg" + method +" does not exist")

        
        np.save('models/' + args.subfolder_path + args.file_name + '_saved_latents_for_' + method,latents) 

        print(sq_diff)

        return imgs, sq_diff, critics

##########################################################################
##########################################################################


def prepare_GAN_nets(sess):
    with open( model, 'rb' ) as file:
        G, D, Gs = pickle.load( file )


    # Take out the variables that correspond to the minibatch standard deviation and set them to zero
    #print([v for v in tf.global_variables()])
    D44ConvLayer = [v for v in tf.global_variables() if v.name == "D_paper/4x4/Conv/weight:0"][0]
    D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

    D44ConvLayer_woMiniBatchStd = sess.run( D44ConvLayer_killMiniBatchStd)
    tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )

    return G,D





##########################################################################
##########################################################################


def prepare_VGG_layers(sess):
    
    K.set_session(sess)
    
    vgg = VGG19(weights='imagenet', include_top=False)
    
    class Prepro(keras.layers.Layer):
        """vgg 19 preprocessing"""

        def __init__(self, units=32, input_dim=(224,224,3)):
            super(Prepro, self).__init__()
        
        def call(self, inputs):
            return preprocess_input(inputs)
    
    preproLayer=Prepro()
        
    vgg_block1_conv2 = keras.Sequential([preproLayer]+vgg.layers[:3])
    vgg_block2_conv2 = keras.Sequential([preproLayer]+vgg.layers[:6])
    vgg_block3_conv2 = keras.Sequential([preproLayer]+vgg.layers[:9])
    vgg_block4_conv4 = keras.Sequential([preproLayer]+vgg.layers[:16])
    vgg_block5_conv4 = keras.Sequential([preproLayer]+vgg.layers[:21])
    
    
    return vgg_block1_conv2, vgg_block2_conv2, vgg_block3_conv2, vgg_block4_conv4, vgg_block5_conv4
    
