"""
Simple matrix intensity plot, similar to MATLAB imagesc()

David Andrzejewski (david.andrzej@gmail.com)
"""

import numpy as NP
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.cm as CM

def scaledimage(W, pixwidth=1, ax=None, grayscale=True):
    """
    Do intensity plot, similar to MATLAB imagesc()
    
    W = intensity matrix to visualize
    pixwidth = size of each W element
    ax = matplotlib Axes to draw on
    grayscale = use grayscale color map
    
    Rely on caller to .show()
    """
    
    # N = rows, M = column
    (N, M) = W.shape
    # Need to create a new Axes?
    if(ax == None):
        ax = P.figure().gca()
    # extents = Left Right Bottom Top
    exts = (0, pixwidth * M, 0, pixwidth * N)
    if(grayscale):
        ax.imshow(W,
                  interpolation='nearest',
                  cmap=CM.gray,
                  extent=exts)
    else:
        ax.imshow(W,
                  interpolation='nearest',
                  extent=exts)

    ax.xaxis.set_major_locator(MT.NullLocator())
    ax.yaxis.set_major_locator(MT.NullLocator())
    return ax

if __name__ == '__main__':
    # Define a synthetic test dataset
    testweights = NP.array([[0.25, 0.50, 0.25, 0.00],
                            [0.00, 0.50, 0.00, 0.00],
                            [0.00, 0.10, 0.10, 0.00],
                            [0.00, 0.00, 0.25, 0.75]])
    # Display it
    ax = scaledimage(testweights)
    P.show()