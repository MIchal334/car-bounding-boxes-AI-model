import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_squer(bounding_box,img):
    print(bounding_box)
    x = [int(bounding_box[0]),int(bounding_box[0]),int(bounding_box[2]),int(bounding_box[2])]
    y = [int(bounding_box[1]),int(bounding_box[3]),int(bounding_box[3]),int(bounding_box[1])]
    fig, ax = plt.subplots()
    ax.imshow(img[0])
    rectangle = patches.Polygon(np.column_stack((x, y)), closed=True, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.title('Wyznaczone auto')
    plt.show()



