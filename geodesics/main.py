import geodesics.utils as utils
from geodesics.find_geodesics import find_geodesics
from geodesics.plotting import *
from geodesics.configs import *


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"


latent_start, latent_end = utils.initialize_endpoints_of_curve( start_end_init)


geodesics_dict = find_geodesics(latent_start, latent_end, methods)

# returns a dictionary of results
# key = methods: linear, Jacobian, proposed
# value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value

for method in methods:
    print(method)
    geodesic_imgs, cost = geodesics_dict[method]
    print(cost)

    #plot_geodesic(geodesic_imgs, method)

plot_geodesic_comparison(geodesics_dict)
