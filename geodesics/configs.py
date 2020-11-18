model = 'models/network-snapshot-008640.pkl'
#model = "models/karras2018iclr-celebahq-1024x1024.pkl" # does not work for some reason

start_end_init="random" # initialization of start and end point

#methods = ["linear","proposed"]
#methods = ["linear"]
#methods = ["Jacobian"]
#methods = ["proposed"]
#methods = ["linear","Jacobian"]
#methods = ["linear","linear"]
#methods = ["proposed","Jacobian"]
methods = ["linear", "Jacobian", "proposed"]



dim_latent = 512 # dimension of latent space
latent_min, latent_max=-2.0, 2.0


no_pts_on_geodesic = 5 # number of interpolation points on curve in latent space with image points in sample space
polynomial_degree = 2 # polynomial degree of parameterized curve in latent space

hyper_critic_penalty=1.0


coefficient_init = 1 # x such that random initialization of poynomial coefficients parameterizing curve between [-x,x]

geodesic_training_steps =30 # number of geodesic training steps
geodesic_learning_rate, adam_beta1, adam_beta2 = 1e-1, 0.5, 0.999


max_critic_value_found = -426.19928
min_critic_value_found = -1117.1604

offset =  -min_critic_value_found + 10
scaling = 1./(max_critic_value_found - min_critic_value_found+ 10)
#offset = 400.0 # discriminator offset to ensure positive scores - should be adaptable
