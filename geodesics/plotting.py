import numpy as np
import PIL.Image


def plot_geodesic(imgs, method):

	imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
	imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
	for idx in range( imgsPlot.shape[0] ):
		PIL.Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/%s_path%d.png' % (method, idx) )

	return None

def plot_geodesic_comparison(geodesics_dict):

	imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
	imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC


	for method in methods:

		imgs = geodesics_dict[method]

		imgs = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
		imgs = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC


		dst = Image.new('RGB', (1024 * no_pts_on_geodesic, 1024))
		
		for k in range(no_pts_on_geodesic):
			

			img_plot = PIL.Image.fromarray( imgs[k], 'RGB' )
			
			dst.paste(img_plot, (1024*k, 0))

			
		dst.save(method + '_path_all')
	# for idx in range( imgsPlot.shape[0] ):
	#     PIL.Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/%s_path%d.png' % (method, idx) )

	# return None
	return None
