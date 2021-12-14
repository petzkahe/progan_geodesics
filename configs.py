import os
import argparse
import sys
from datetime import datetime
import numpy as np


##########################################################################################################################################
##########################################################################################################################################
### CONFIGURATIONS START ###
##########################################################################################################################################
##########################################################################################################################################



#############################
## General configurations
#############################
global_seed = 1000

model = './models/CelebA/karras2018iclr-celebahq-1024x1024.pkl'  # Path to models, pickled as G, D, GS

methods = ["linear","linear_in_sample", "disc", "sqDiff", "sqDiff+D", "vgg", "vgg+D"] # Methods to find geodesics for

gpu_id = '3' # GPU unit to use for training

experiment_id = "my_experiment" # Specifies the folder in which results will be saved
optional_run_id = "" # Optional run_id for several experiments with the same experiment_id

COMPUTE_STATISTICS_OFF = False  # If this default flag is False, then discriminator statistics are computed and  the flag can be used in command line to turn it off
TRAIN_GEODESICS_OFF = False    # If this default flag is False, then geodesics are learned and  the flag can be used in command line to turn it off
VIDEO_GENERATION_OFF = False   # If this default flag is False, then videos are generated and the flag can be used in command line to turn it off
START_SEED_OFF = False # Option to switch from seed to loading starting point in latent space
END_SEED_OFF = False # Option to switch from seed to loading end point in latent space


#############################
## Geodesic configurations:
#############################
start = 477 # random seed for starting point of geodesic
end = 56  #  random seed for end point of geodesic

n_pts_on_geodesic = 10 # number of interpolation points on curve in latent space with image points in sample space
polynomial_degree = 3  # polynomial degree of parameterized curve in latent space

n_training_steps =300 # number of geodesic training steps
learning_rate = 1e-3 # learning rate for optimization of geodesics
adam_beta1, adam_beta2 = 0.5, 0.999 ## Adam parameters for optimization of geodesics

hyper_sqDiff = 0.2  # for "method mse_plus_disc": trade-off between mse and discriminator penalty used 
hyper_vgg = 0.2  # for "vgg_plus_disc": trade-off between vgg dists and discriminator penalty
coefficient_init = .0001 # x s.t. random initialization of poynomial coefficients parameterizing curve between [-x,x], [the smaller this value, the closer we start to linear curve in latent space]

SPHERICAL_INTERPOLATION_ON = False # Turn off spherical interpolation pattern

#############################
## Statistics configurations:
#############################

n_statistics= 5000 # Number of samples to calculate statistics, has to be larger than 50
tfrecord_dir = './dataset/CelebA/tfrecords' # Reference to dataset of real images in tfrecord format

#############################
## Video configurations:
#############################

n_frames = 100 # number of frames composing the video
fps = 15.0  # frames per second 
video_percentage = 100 # option to not use the entire path for video, if <100 then rest of video cut off, also saves numpy file for latent point at cut off



#############################
## Dataset configurations:
#############################
dim_latent = 512 # dimension of latent space


###############################
# Discriminator Usage configurations:
# (tries to load it from a previous calculation of statistics, see below)
###############################

max_critic_value = 130
min_critic_value = -1400 
max_ideal_critic_value = max_critic_value   # Geodesic points with critic value above this value do not get penalized


##########################################################################################################################################
##########################################################################################################################################
### CONFIGURATIONS END ###
##########################################################################################################################################
##########################################################################################################################################


################################
# Arg parsing code for manipulation from command line
###############################

def configure():
  

  print("\n \n-------------\nSetup \n-------------")

  parser = argparse.ArgumentParser(description='Find GAN paths')
    
  parser.add_argument("--global_seed", help="", default=global_seed, type=int)
  parser.add_argument("--gpu_id", help="", default=gpu_id, type=str)
  parser.add_argument("--model", help="", default=model, type=str)
  
  
  parser.add_argument("--experiment_id", help="", default=experiment_id, type=str)
  parser.add_argument("--optional_run_id", help="", default=optional_run_id, type=str)
  
  parser.add_argument("--COMPUTE_STATISTICS_OFF", help="", default=COMPUTE_STATISTICS_OFF, action = 'store_true')
  parser.add_argument("--TRAIN_GEODESICS_OFF", help="", default=TRAIN_GEODESICS_OFF, action = 'store_true')
  parser.add_argument("--VIDEO_GENERATION_OFF", help="", default=VIDEO_GENERATION_OFF, action = 'store_true')
  parser.add_argument("--START_SEED_OFF", help="", default=START_SEED_OFF, action = 'store_true')
  parser.add_argument("--END_SEED_OFF", help="", default=END_SEED_OFF, action = 'store_true')
  
  parser.add_argument("--start", help="", default=start, type=int)
  parser.add_argument("--end", help="", default=end, type=int)  
  
  parser.add_argument("--n_pts_on_geodesic", help="", default=n_pts_on_geodesic, type=int)
  parser.add_argument("--polynomial_degree", help="", default=polynomial_degree, type=int)
  parser.add_argument("--n_training_steps", help="", default=n_training_steps, type=int)
  parser.add_argument("--learning_rate", help="", default=learning_rate, type=float)
  parser.add_argument("--SPHERICAL_INTERPOLATION_OFF", help="", default=SPHERICAL_INTERPOLATION_ON, action = 'store_true')
  
  parser.add_argument("--adam_beta1", help="", default=adam_beta1, type=float)
  parser.add_argument("--adam_beta2", help="", default=adam_beta2, type=float)
  
  parser.add_argument("--hyper_sqDiff", help="", default=hyper_sqDiff, type=float)
  parser.add_argument("--hyper_vgg", help="", default=hyper_vgg, type=float)
  parser.add_argument("--coefficient_init", help="", default=coefficient_init, type=float)
  
  parser.add_argument("--n_statistics", help="", default=n_statistics, type=int)
  parser.add_argument("--tfrecord_dir", help="", default=tfrecord_dir, type=str)  
  
  parser.add_argument("--n_frames", help="", default=n_frames, type=int)
  parser.add_argument("--fps", help="", default=fps, type=float)
  parser.add_argument("--video_percentage", help="", default=video_percentage, type=float)
  
  parser.add_argument("--max_critic_value", help="", default=max_critic_value, type=float)
  parser.add_argument("--min_critic_value", help="", default=min_critic_value, type=float)
  parser.add_argument("--max_ideal_critic_value", help="", default=max_ideal_critic_value, type=int)
  
  parser.add_argument("--dim_latent", help="", default=dim_latent, type=int)

  parser.add_argument("--methods", help="", default=methods, type=str, nargs='+')

  args = parser.parse_args()
  
  configurations = {} 
  original_stdout = sys.stdout # Save a reference to the original standard output
  
  try:
    max_critic_value_new = np.load("results/"+getattr(args,'experiment_id')+"/statistics/max_min.npy")[0]   # Maximal critic value on fake data
    min_critic_value_new = np.load("results/"+getattr(args,'experiment_id')+"/statistics/max_min.npy")[1]   # Minimal critic value on fake data
    max_ideal_critic_value_new = max_critic_value_new   # Geodesic points with critic value above this value do not get penalized
    print("Maximal and minimal critic values found on file.")
    setattr(args,'max_critic_value',max_critic_value_new)
    setattr(args,'min_critic_value',min_critic_value_new)
    setattr(args,'max_ideal_critic_value',max_ideal_critic_value_new)
  except FileNotFoundError:
    print("No maximal and minimal critic values found on file.")
      
  for arg in vars(args):
    configurations[arg] = getattr(args,arg)
    print(arg, getattr(args,arg))
  
    
  # CREATE RUN FOLDERS
  if not os.path.exists('results/'+configurations['experiment_id']+'/images/'):
    os.makedirs('results/'+configurations['experiment_id']+'/images/')

  if not os.path.exists('results/'+configurations['experiment_id']+'/coefficients/'):
    os.makedirs('results/'+configurations['experiment_id']+'/coefficients/')
    
  if not os.path.exists("results/"+configurations['experiment_id']+"/statistics/"):
    os.makedirs("results/"+configurations['experiment_id']+"/statistics/")
    
  if not os.path.exists("results/"+configurations['experiment_id']+"/videos/tmp/"):
    os.makedirs("results/"+configurations['experiment_id']+"/videos/tmp/")
    
  assert(configurations['n_statistics']>99), "ERROR --n_statistics must be larger than 100 to produce the sample grid"
    
      
  with open('results/'+configurations['experiment_id']+'/setting_'+configurations['optional_run_id']+'.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d / %H:%M:%S")
    print("Logs from run on: ", dt_string)
    print("\nArguments used:\n---------------------")
    for arg in vars(args):
      configurations[arg] = getattr(args,arg)
      print(arg, getattr(args,arg))
    
    sys.stdout = original_stdout # Reset the standard output to its original value 
    
  


  print('')
  
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
  os.environ["CUDA_VISIBLE_DEVICES"]=configurations['gpu_id']

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
  # 0 = all messages are logged (default behavior)
  # 1 = INFO messages are not printed
  # 2 = INFO and WARNING messages are not printed
  # 3 = INFO, WARNING, and ERROR messages are not printed



  print("-------------")
  print("Setup success\n")
  print("-------------")
  print("")
  
  return configurations 






