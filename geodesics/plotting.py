import numpy as np
import PIL.Image as Image
from geodesics.configs import *
import matplotlib.pyplot as plt
import os


def plot_geodesic(imgs, method):

	imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
	imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

	if not os.path.exists('images/singles'):
		os.makedirs('images/singles')

	for idx in range( imgsPlot.shape[0] ):
		Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/singles/%s_path%d.png' % (args.folder_name, method, idx) )

	return None

def plot_geodesic_comparison(geodesics_dict):

	dst = Image.new('RGB', (1024 * no_pts_on_geodesic, 1024 * len(methods)))
	
	for k_method in range(len(methods)):

		method = methods[k_method]
		
		[imgs,cost,critics] = geodesics_dict[method]

		imgs = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
		imgs = imgs.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

		for k_path in range(no_pts_on_geodesic):
	
			img_plot = Image.fromarray( imgs[k_path], 'RGB' )
			
			dst.paste(img_plot, (1024*k_path, 1024*k_method))


	size = int(dst.width/4),int(dst.height/4)
	dst_small = dst.resize(size,resample=Image.BILINEAR)	
	
	#sep = '&' #sep.join(methods)
	dst_small.save( './images/%spaths_%s.png' % (args.subfolder_path, args.file_name) ) 
	# for idx in range( imgsPlot.shape[0] ):
	#     Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/%s_path%d.png' % (method, idx) )

	# return None
	return None

def plot_critics(geodesics_dict):
	k = 0
	color_marker = ['rd-','k+-','bv-','g^-','yo-','ms-','k*-']

	for method in methods:

		[_, _, critics] = geodesics_dict[method]
		critics_plot = [item for sublist in critics for item in sublist]
		plt.plot(range(len(critics_plot)),critics_plot,color_marker[k],label=method)
		k = k + 1

	plt.ylabel('Critic value')
	plt.legend()
	plt.savefig('images/%scritics_%s' % (args.subfolder_path, args.file_name) )


	return None




