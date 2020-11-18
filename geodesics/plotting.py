import numpy as np
import PIL.Image


def plot_geodesic(imgs, method):

    imgsPlot = np.clip( np.rint( (imgs + 1.0) / 2.0 * 255.0 ), 0.0, 255.0 ).astype( np.uint8 )  # [-1,1] => [0,255]
    imgsPlot = imgsPlot.transpose( 0, 2, 3, 1 )  # NCHW => NHWC
    for idx in range( imgsPlot.shape[0] ):
        PIL.Image.fromarray( imgsPlot[idx], 'RGB' ).save( './images/%s_path%d.png' % (method, idx) )

    return None
