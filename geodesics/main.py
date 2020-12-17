import geodesics.utils as utils
from geodesics.find_geodesics import find_geodesics
from geodesics.plotting import *
from geodesics.configs import *


import argparse




def main(args=None):

    latent_start, latent_end = utils.initialize_endpoints_of_curve( start_end_init)


    geodesics_dict = find_geodesics(latent_start, latent_end, methods)

    # returns a dictionary of results
    # key = methods: linear, mse, mse_plus_disc
    # value =  a list of two things: curves_in_latent_space_value, curves_in_sample_space_value

    for method in methods:
        print(method)
        geodesic_imgs, cost, critics = geodesics_dict[method]
        print("Squared differences:")
        print(cost)
        print("Sum: " + str(np.sum(cost)))
        print("Critic values:")
        print(critics)

        #plot_geodesic(geodesic_imgs, method)

    plot_geodesic_comparison(geodesics_dict)
    plot_critics(geodesics_dict)

    return None


if __name__ == "__main__":
    
    main(args)

