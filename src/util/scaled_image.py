"""
simple matrix intensity plot, similar to MATLAB imagesc()

@author: David Andrzejewski (david.andrzej@gmail.com)
"""

import matplotlib, numpy;

"""
do intensity plot, similar to MATLAB imagesc()
still rely on caller to .show()

@param W: intensity matrix to visualize
@param pixel_width: size of each W element
@param axes: matplotlib Axes to draw on 
@param gray_scale: use grayscale color map
"""
def scaled_image(W, pixel_width=1, axes=None, gray_scale=True):

    # N = rows, M = column
    (N, M) = W.shape 
    # need to create a new Axes?
    if(axes == None):
        axes = P.figure().gca()
    # extents = Left Right Bottom Top
    exts = (0, pixel_width * M, 0, pixel_width * N)
    if(gray_scale):
        axes.imshow(W, interpolation='nearest', cmap=matplotlib.cm.gray, extent=exts)
    else:
        axes.imshow(W, interpolation='nearest', extent=exts)

    axes.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    axes.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    return axes

if __name__ == '__main__':
    # define a synthetic test dataset
    testweights = numpy.array([[0.25, 0.50, 0.25, 0.00],
                            [0.00, 0.50, 0.00, 0.00],
                            [0.00, 0.10, 0.10, 0.00],
                            [0.00, 0.00, 0.25, 0.75]])
    
    # display it
    ax = scaled_image(testweights)
    matplotlib.pyplot.show()
