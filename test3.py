import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
import matplotlib.ticker as ticker
from PIL import Image

"""
# Method 1
imageFile = 'final_result.png'
#imageFile = 'segmented_lungs.png'
mat = imread(imageFile)

mat = mat[:, :, 0]  # get the first channel
rows, cols = mat.shape
xv, yv = np.meshgrid(range(cols), range(rows)[::-1])

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(projection='3d')
ax.elev = 75
ax.plot_surface(xv, yv, mat, cmap='gray', linewidth=0)
plt.show()

plt.cm.jet
"""

# Method 2
def ImPlot2D3D(img, cmap="gray", step=False, ratio=10):

    if step:
        img = (img.repeat(ratio, axis=0)).repeat(ratio, axis=1)

    Z = img[::1, ::1]

    fig = plt.figure(figsize=(14, 7))

    # 2D Plot
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(Z, cmap=cmap)
    ax1.set_title('2D')
    ax1.grid(False)

    # 3D Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    X, Y = np.mgrid[:Z.shape[0], :Z.shape[1]]
    ax2.plot_surface(X, Y, Z, cmap=cmap)
    ax2.set_title('3D')

    # Scale the ticks back down to original values
    if step:
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / ratio))
        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / ratio))
        ax1.xaxis.set_major_formatter(ticks_x)
        ax1.yaxis.set_major_formatter(ticks_y)
        ax2.xaxis.set_major_formatter(ticks_x)
        ax2.yaxis.set_major_formatter(ticks_y)

    plt.show()


imageFile = 'final_result.png'
mat = imread(imageFile)
mat = mat[:, :, 0]
arr = np.array(mat)
ImPlot2D3D(arr)
