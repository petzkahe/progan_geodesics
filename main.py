from find_geodesics import find_geodesics
from utils import *
from configs import *
from statistics import *


def main():

    # Configuration loads all the parameters from the config file and overwrites
    # parameters that are specified on the command line
    
    configurations = configure()
    
    ####################################
    # Compute statistics of critic values
    ####################################
    if configurations['COMPUTE_STATISTICS_OFF'] == False:
      
      # Compute range of critic values
       
      maxim, minim = discriminator_value_statistics(configurations['model'], configurations['global_seed'], configurations['n_statistics'], configurations['gpu_id'], configurations['experiment_id'], configurations['optional_run_id'], configurations['tfrecord_dir'])
      
      configurations['max_critic_value'] = maxim
      configurations['min_critic_value'] = minim
      configurations['max_ideal_critic_value'] = maxim
      
      print("Maximal and minimal critic values changed from default to new values.")
            
      print("Plotting of histograms of discriminator values and sample grids completed!\n---------------------\n")

  
     
  
  
  
    ####################################
    # Training the geodesics
    ####################################
    
    if configurations['TRAIN_GEODESICS_OFF'] == False:    
      
      # The following returns a dictionary of results
      # key = methods: linear, mse, mse_plus_disc, vgg, vgg_plus_disc
      # value =  a list of two things:  curves_in_sample_space and critic values along path

      geodesics_dict = find_geodesics(**configurations)
    
      print("\nPlotting the results\n")
      plot_geodesic_comparison(geodesics_dict, configurations['methods'], configurations['n_pts_on_geodesic'], configurations['experiment_id'], configurations['optional_run_id'])
      plot_critics(geodesics_dict, configurations['methods'], configurations['experiment_id'], configurations['optional_run_id'])
      plot_sqDiff(geodesics_dict, configurations['methods'], configurations['experiment_id'], configurations['optional_run_id'])



    ####################################
    # Video generation
    ####################################
    
    if configurations['VIDEO_GENERATION_OFF'] == False:
      
      video_generation(**configurations)
    
    print("Successful!\n")

    return None



if __name__ == "__main__":
    
    main()

