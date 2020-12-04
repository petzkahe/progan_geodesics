import tensorflow as tf
import pickle
import numpy as np
import geodesics.tfutil as tfutil
import geodesics.graph as graph
from geodesics.configs import *



##########################################################################
##########################################################################


def find_geodesics(latent_start, latent_end, methods):
    geodesics_dict = {}

    for method in methods:
        with tf.Session() as sess:

            G, D = prepare_GAN_nets( sess )

            geodesics_dict[method] = learn_geodesic(method, latent_start, latent_end, sess, G, D)

        tf.reset_default_graph()
        sess.close()



    return geodesics_dict




##########################################################################
##########################################################################


def learn_geodesic(method, latent_start, latent_end, sess, G, D):



        ################################################################################################################################################

        if method=="linear":

            # A disadvantage of loading the graph by a method could be that it does not import lazily but goes through all lines of the method
            images, squared_differences, latent_plchldr, labels_plchldr = graph.import_linear_graph(G,D)


            latents = graph.parameterize_line(latent_start, latent_end) # is of size (no_pts, dimension)
            print("Linear latents norms:")
            print(np.linalg.norm(latents,axis=1))

            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )

            [imgs, cost] = tf.get_default_session().run( [images, squared_differences], feed_dict={latent_plchldr: latents, labels_plchldr: lbls} )

        ################################################################################################################################################

        elif method == "Jacobian":

            # coefficients_free are the variables to learn, which are coefficients of the interpolating polynomial
            # latent_tensor contains the latent points on the curve
            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)

            images, squared_differences, objective_Jacobian, latent_plchldr, labels_plchldr = graph.import_Jacobian_graph( G, D , latents_tensor)

            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("training_Jacobian"):
                    train_Jacobian = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective_Jacobian,
                    var_list=coefficients_free
                )


            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_Jacobian')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )



            for iteration in range( geodesic_training_steps ):
                _, x = sess.run( [train_Jacobian, objective_Jacobian], feed_dict={labels_plchldr: lbls} )

                print(x)
                if iteration % 500 == 0:
                    print( str( int( iteration /geodesic_training_steps * 100.0 ) ) + ' %' )


            latents = sess.run(latents_tensor)


            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )

            # Why not use sess?
            #[imgs, cost] = tf.get_default_session().run( [images, squared_differences],
            #                                             feed_dict={latent_plchldr: latents, labels_plchldr: lbls} )
            [imgs, cost] = sess.run( [images, squared_differences],
                                                         feed_dict={latent_plchldr: latents, labels_plchldr: lbls} )







################################################################################################################################################

        elif method == "proposed":

            latents_tensor, coefficients_free = graph.parameterize_curve(latent_start, latent_end)
            images, squared_differences, objective_proposed, latent_plchldr, labels_plchldr, critic_objective = graph.import_proposed_graph( G, D , latents_tensor)

            sess.run(tf.variables_initializer([coefficients_free]))

            with tf.variable_scope("training_proposed"):
                    train_proposed = tf.train.AdamOptimizer(
                    learning_rate=geodesic_learning_rate,
                    beta1=adam_beta1,
                    beta2=adam_beta2
                ).minimize(
                    objective_proposed,
                    var_list=coefficients_free
                )

            adam_training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_proposed')
            sess.run( tf.variables_initializer( adam_training_variables ) )

            lbls = np.zeros( [latents_tensor.shape[0]] + G.input_shapes[1][1:] )



            for iteration in range( geodesic_training_steps ):
                _, x, sqrdiff,critic_obj = sess.run( [train_proposed, objective_proposed, squared_differences, critic_objective], feed_dict={labels_plchldr: lbls} )

                print(x)
                if iteration % 500 == 0:
                    print( str( int( iteration /geodesic_training_steps * 100.0 ) ) + ' %' )


            latents = sess.run(latents_tensor)
            print("Proposed latents norms:")
            print(np.linalg.norm(latents,axis=1))


            lbls = np.zeros( [latents.shape[0]] + G.input_shapes[1][1:] )

            [imgs, cost] = tf.get_default_session().run( [images, squared_differences],
                                                         feed_dict={latent_plchldr: latents, labels_plchldr: lbls} )



        else:
             raise Exception("Method" + method +" does not exist")

        np.save('models/saved_latents_' + method,latents) 

        return imgs, cost

##########################################################################
##########################################################################


def prepare_GAN_nets(sess):
    with open( model, 'rb' ) as file:
        G, D, Gs = pickle.load( file )


    # Take out the variables that correspond to the minibatch standard deviation and set them to zero
    #D44ConvLayer = [v for v in tf.global_variables() if v.name == "D/4x4/Conv/weight:0"][0]
    #D44ConvLayer_killMiniBatchStd = D44ConvLayer[:, :, 512, :].assign( tf.zeros( (3, 3, 512) ) )

    #D44ConvLayer_woMiniBatchStd = sess.run( D44ConvLayer_killMiniBatchStd)
    #tfutil.set_vars( {D44ConvLayer: D44ConvLayer_woMiniBatchStd} )

    return G,D



##########################################################################
##########################################################################
