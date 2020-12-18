import os
import argparse
import sys
from datetime import datetime


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args():

    parser = argparse.ArgumentParser(description='Find GAN paths')
    
    parser.add_argument("--startpoint_seed", help="", default=200, type=int)
    parser.add_argument("--endpoint_seed", help="", default=250, type=int)
    parser.add_argument("--no_pts_on_geodesic", help="", default=10, type=int)
    parser.add_argument("--polynomial_degree", help="", default=3, type=int)
    parser.add_argument("--hyperparam_disc_vs_mse", help="", default=1, type=float)
    parser.add_argument("--hyperparam_disc_vs_vgg", help="", default=1e6, type=float)
    parser.add_argument("--coefficient_init", help="", default=0.001, type=float)
    parser.add_argument("--geodesic_training_steps", help="", default=1000, type=int)
    parser.add_argument("--geodesic_learning_rate", help="", default=1e-4, type=float)
    parser.add_argument("--gpu_id", help="", default="3", type=str)
    parser.add_argument("--PATH", help="", default="~/github/CelebAgeodesics/", type=str)
    parser.add_argument("--subfolder_path", help="", default="default_run/", type=str)
    parser.add_argument("--file_name", help="", default="", type=str)
    parser.add_argument("--use_objective_from_paper", help="", default=True, type=boolean_string)
    parser.add_argument("--methods", help="", default=["linear","linear_in_sample", "disc", "mse", "mse_plus_disc", "vgg", "vgg_plus_disc"], type=str, nargs='*')


    args = parser.parse_args()

    return args




args = get_args()
for arg in vars(args):
    print(arg, getattr(args,arg))
print('')



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


# CREATE RUN FOLDERS
if not os.path.exists('images/' + args.subfolder_path):
    os.makedirs('images/' + args.subfolder_path)

if not os.path.exists('models/' + args.subfolder_path):
    os.makedirs('models/' + args.subfolder_path)


# PRINT ARGUMENTS TO FILE
original_stdout = sys.stdout # Save a reference to the original standard output

with open('./images/%sargs_%s.txt' % (args.subfolder_path, args.file_name), 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d / %H:%M:%S")
    print("Logs from run on: ", dt_string)
    print("\nArguments used:\n---------------------")
    for arg in vars(args):
        print(arg, getattr(args,arg))
    print("---------------------")
    sys.stdout = original_stdout # Reset the standard output to its original value


# SOME LEFTOVER HARDCODED ARGS
model = 'models/karras2018iclr-celebahq-1024x1024.pkl'
#model = "models/karras2018iclr-celebahq-1024x1024.pkl" # does not work for some reason
start_end_init= "random" # initialization of start and end point
dim_latent = 512 # dimension of latent space
adam_beta1, adam_beta2 = 0.5, 0.999

# For final 1024x CelebA model:
max_critic_value_found = -230
min_critic_value_found = -526
max_ideal_critic_value = -230 #-300
offset =  min_critic_value_found - 10
scaling = 1./(max_ideal_critic_value - offset)


# ASSIGN HARDCODED ARGS FROM ARGPARSER NAMESPACE -> to be replaced with args.<arg> from argparser

methods = args.methods
use_objective_from_paper = args.use_objective_from_paper
startpoint_seed = args.startpoint_seed  # 200
endpoint_seed = args.endpoint_seed # 250 fairly ok endpoint images #300 #547
no_pts_on_geodesic = args.no_pts_on_geodesic # number of interpolation points on curve in latent space with image points in sample space
polynomial_degree = args.polynomial_degree # polynomial degree of parameterized curve in latent space
hyperparam_disc_vs_mse = args.hyperparam_disc_vs_mse #.1 # trade-off between mse and discriminator penalty
hyperparam_disc_vs_vgg = args.hyperparam_disc_vs_vgg #.1 # trade-off between vgg dists and discriminator penalty
coefficient_init = args.coefficient_init # .0001 # x such that random initialization of poynomial coefficients parameterizing curve between [-x,x]
geodesic_training_steps = args.geodesic_training_steps # 100# number of geodesic training steps
geodesic_learning_rate = args.geodesic_learning_rate









