import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('segmented_lungs.png', 0)
mask = cv2.imread('masked_lungs.png', 0)

# flood fill to remove mask at borders of the image
h, w = img.shape[:2]
for row in range(h):
    if mask[row, 0] == 255:
        cv2.floodFill(mask, None, (0, row), 0)
    if mask[row, w-1] == 255:
        cv2.floodFill(mask, None, (w-1, row), 0)
for col in range(w):
    if mask[0, col] == 255:
        cv2.floodFill(mask, None, (col, 0), 0)
    if mask[h-1, col] == 255:
        cv2.floodFill(mask, None, (col, h-1), 0)

# flood fill background to find inner holes
holes = mask.copy()
cv2.floodFill(holes, None, (0, 0), 255)

# invert holes mask, bitwise or with mask to fill in holes
holes = cv2.bitwise_not(holes)
mask = cv2.bitwise_or(mask, holes)

# display masked image
masked_img = cv2.bitwise_and(img, img, mask=mask)
masked_img_with_alpha = cv2.merge([img, img, img, mask])

plt.imshow(masked_img_with_alpha)
plt.show()

blurred = cv2.GaussianBlur(masked_img_with_alpha, (3, 3), 0)

edges = cv2.Canny(blurred, 100, 200)

again_masked_img_with_alpha = cv2.merge([edges, edges, edges, mask])

# show it
plt.imshow(again_masked_img_with_alpha, cmap="gray")
plt.show()

img_gray = cv2.cvtColor(again_masked_img_with_alpha, cv2.COLOR_RGB2GRAY)

# Find the contour of the figure
image, contours, hierarchy = cv2.findContours(
                                   image = img_gray,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours
contours = sorted(contours, key = cv2.contourArea, reverse = True)

print("Number of Contours is: " + str(len(contours)))

# Draw the contour
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)

plt.axis('off')
plt.imshow(img_copy, cmap="gray")
plt.savefig("final_result", bbox_inches='tight', pad_inches=0)
plt.show()
