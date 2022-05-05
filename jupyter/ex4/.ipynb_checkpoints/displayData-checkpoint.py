import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# displays a random selection of 100 examples in a 10 x 10 grid

def displayData(Xselection, noGridColumns):
    # calculate the number of rows
    noGridRows = Xselection.shape[0] // noGridColumns
    
    # convert the input data into a datastructure that works for the grid part:
    # an array of 20 x 20 matrices, each matrix representing a single example
    images = []
    for i in range(Xselection.shape[0]):
        # single image in a 400 x 1 matrix
        single_raw = Xselection[i, :, None]
        # reshape to a 20 x 20 matrix
        single = np.reshape(single_raw,(20,20)).T
        images.append(single)


    # https://kanoki.org/2021/05/11/show-images-in-grid-inside-jupyter-notebook-using-matplotlib-and-numpy/

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(noGridRows, noGridColumns),  # creates grid of axes
                     axes_pad=0.1,  # pad between axes
                     )

    for ax, im in zip(grid, images):
        ax.imshow(im, cmap='gray')

    plt.show()