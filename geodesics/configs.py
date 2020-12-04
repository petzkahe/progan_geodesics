import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#model = 'models/network-snapshot-008640.pkl'
model = 'models/karras2018iclr-celebahq-1024x1024.pkl'
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

endpoint_seed = 250 #300 #547

dim_latent = 512 # dimension of latent space
latent_min, latent_max=-2.0, 2.0


no_pts_on_geodesic = 10 # number of interpolation points on curve in latent space with image points in sample space
polynomial_degree = 2 # polynomial degree of parameterized curve in latent space

hyper_critic_penalty=.05 # trade-off between squared dists and distriminator function


coefficient_init = .1 # x such that random initialization of poynomial coefficients parameterizing curve between [-x,x]

geodesic_training_steps = 1000 # number of geodesic training steps
geodesic_learning_rate, adam_beta1, adam_beta2 = 1e-1, 0.5, 0.999

# Old mid-training model
#max_critic_value_found = -426.19928
#min_critic_value_found = -1117.1604

# Final 1024x CelebA model:
max_critic_value_found = 189
min_critic_value_found = -168

offset =  - (min_critic_value_found - 10)
scaling = 1./(max_critic_value_found - min_critic_value_found+ 10)
#offset = 400.0 # discriminator offset to ensure positive scores - should be adaptable
