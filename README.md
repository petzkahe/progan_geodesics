# CelebAgeodesics

--------------------------
DESCRIPTION:
  The code allows to calculate statisics for the discriminator, find geodesics and create videos from the geodesics

--------------------------
RUNNING THE CODE:

  The main file should be run from the folder "./geodesicsProgan" with the command: 

    python main.py

  Alternatively one can set up and run bash scripts from the same location with
  
  bash bashScripts/name_of_bash_script

---------------------------
CONFIGURATIONS:

  Current default values for configurations can be checked and adjusted in the beginning of the "configs.py" file before CONFIGURATIONS END

  All default values can be overwritten from the command line by adding the correspoding flag.
  For example, to change the experiment_id and add an optional run_id, use the command:

    python main.py --run_id="my_new_experiment" --optional_run_id="test_1"

  This will save the results in the folder "results/my_experiment" and all file names have a suffix of "test_1"

  By default, the code calculates discriminator statistics, trains geodesics and creates a video.
  Each of these steps can be skipped by using the flags:
    COMPUTE_STATISTICS_OFF
    TRAIN_GEODESICS_OFF 
    VIDEO_GENERATION_OFF 

  For example, if you only want to train the geodesics, then use the command

    python main.py --COMPUTE_STATISTICS_OFF --VIDEO_GENERATION_OFF
    
  If you do not want to run all the default methods, then use the command
    
    python main.py --methods "method_1" "method_2" "method_3"
    
  REUSE_OF_DISCRIMINATOR_STATISTICS:     
  
  For each experiment_id, running the critic statistics saves the max and min value of the critic.
  If the same experiment_id is used again, the code loads these values in the following run during configuration, so these values are used if no new critic statistics are calculated. 
  
  This suggests the following usage:
    Never use the same experiment_id for different models
    Turn off critic statistics if the same model and experiment_id have been used before.
  
  For example, if you want to run different start and endpoints for one model, use the commands
  
    python main.py --start=1 --end=2 --experiment_id="my_experiment" --optional_run_id="seeds_1_2"
    python main.py --start=3 --end=4 --experiment_id="my_experiment" --optional_run_id="seeds_3_4" --COMPUTTE_STATISTICS_OFF
    python main.py --start=5 --end=6 --experiment_id="my_experiment" --optional_run_id="seeds_5_6" --COMPUTTE_STATISTICS_OFF
  
  

------------------------------------------
RESULTS:

  Computing critic statistics returns under "results/experiment_id/statistics/"
    A sample grid of real images with critic values
    A sample grid of generated images with critic values
    A sample grid of generated images with the lowest found critic values
    A sample grid of generated images with the highest found critic values
    A histogramm comparing critic values of real and generated samples
    A numpy file (.npy) containing the estimated max and min value over all discriminator values
  
  Training geodesics returns
    Images under "results/experiment_id/images"
      An image containing the geodesics for each method
      An image containing the critic values along the geodesics for each method
      An image containing the square differences along the geodesics for each method
  
    Coefficients of the geodesics are saved under "results/experiment_id/coefficient"

  Video generation returns 
    A video showing all geodesics under "results/experiment_id/videos"
    An image of squared differences along the geodesic under "results/experiment_id/images
    An image of critic values along the geodesic under "results/experiment_id/images    
  
