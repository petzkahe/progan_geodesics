import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import pickle
import PIL.Image
import matplotlib.pyplot as plt

import geodesics.tfutil as tfutil
import numpy as np
import tensorflow as tf
import geodesics.utils as utils


critic_values_load = np.load("models/critic_values_raw.npy")

plt.hist(critic_values_load, bins=40)
plt.savefig("images/hist_of_critic_on_generated"s)


