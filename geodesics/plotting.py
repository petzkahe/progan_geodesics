import numpy as np
import PIL.Image as Image
from geodesics.configs import *


def plot_geodesic(imgs, method):

	imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
	imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
	for idx in range( imgsPlot.shape[0] ):
		Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/singles/%s_path%d.png' % (method, idx) )

	return None

def plot_geodesic_comparison(geodesics_dict):

	dst = Image.new('RGB', (1024 * no_pts_on_geodesic, 1024 * len(methods)))
	
	for k_method in range(len(methods)):

		method = methods[k_method]
		
		[imgs,cost] = geodesics_dict[method]

		imgs = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
		imgs = imgs.transpose( 0, 2, 3, 1 )  # NCHW => NHWC

		for k_path in range(no_pts_on_geodesic):
	
			img_plot = Image.fromarray( imgs[k_path], 'RGB' )
			
			dst.paste(img_plot, (1024*k_path, 1024*k_method))

	size = int(dst.width/4),int(dst.height/4)
	dst_small = dst.resize(size,resample=Image.BILINEAR)	
	sep = ', '
	dst_small.save( './images/path_for_%s.png' % sep.join(methods))
	# for idx in range( imgsPlot.shape[0] ):
	#     Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/%s_path%d.png' % (method, idx) )

	# return None
	return None
