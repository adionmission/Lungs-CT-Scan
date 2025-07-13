import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation
import matplotlib.pyplot as plt
from glob import glob
import pydicom as dicom
import dicom
import dicom_numpy
import os
import cv2

PATH = "D:/CT_scans/resized/"


def extract_voxel_data(path):
    datasets = [cv2.imread(path + "/" + f, 0) for f in os.listdir(path)]

    return datasets


def get_pixels_hu(scans):
    image = np.stack([s for s in scans])

    image = image.astype(np.int16)

    return np.array(image, dtype=np.int16)


test_patient_scans = extract_voxel_data(PATH)
test_patient_images = get_pixels_hu(test_patient_scans)

testt_image = test_patient_images[0]

print(type(testt_image))
print("Original shape: "+str(testt_image.shape))

#-----------------------------------------------------------------------
'''
testt_image = testt_image[:,:,0]

testt_image = cv2.resize(testt_image,(512, 512))

print("After reshape: "+str(testt_image.shape))
'''
#-----------------------------------------------------------------------

IMG_SIZE_1 = testt_image.shape[0]
IMG_SIZE_2 = testt_image.shape[1]

print(IMG_SIZE_1)
print(IMG_SIZE_2)

print("Original Slice")
plt.axis('off')
plt.imshow(testt_image, cmap='gray')
plt.show()


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((IMG_SIZE_1, IMG_SIZE_2), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


# Show some example markers from the middle
test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(testt_image)

'''
print("Internal Marker")
plt.imshow(test_patient_internal, cmap='gray')
plt.show()
print("External Marker")
plt.imshow(test_patient_external, cmap='gray')
plt.show()
print("Watershed Marker")
plt.imshow(test_patient_watershed, cmap='gray')
plt.show()
'''


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((IMG_SIZE_1, IMG_SIZE_2)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


# Some Testcode:
test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = seperate_lungs(
    testt_image)

'''
print("Sobel Gradient")
plt.imshow(test_sobel_gradient, cmap='gray')
plt.show()
print("Watershed Image")
plt.imshow(test_watershed, cmap='gray')
plt.show()
print("Outline after reinclusion")
plt.imshow(test_outline, cmap='gray')
plt.show()
'''
print("Lungfilter after closing")
plt.axis('off')
plt.imshow(test_lungfilter, cmap='gray')
plt.show()

print("Segmented Lung")
plt.axis('off')
plt.imshow(test_segmented, cmap='gray')
plt.show()
