from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
import numpy as np
from glob import glob
from PIL import Image
import cv2 as cv


img = np.array(Image.open('/Users/chengwensun/Library/CloudStorage/GoogleDrive-nikki1231121@gmail.com/Other computers/My Computer/Umass/FALL 2022/ECE 597IP/Project2/over3.png'))


def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)


labels = segmentation.slic(img, compactness=30, n_segments=400)
labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
regions = regionprops(labels)



label_rgb = color.label2rgb(labels, img, kind='avg')

label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))

rag = graph.rag_mean_color(img, labels)

for region in regions:
    rag.nodes[region['label']]['centroid'] = region['centroid']


def display_edges(image, g,):
    """Draw edges of a RAG on its image

    Returns a modified image with the edges drawn.Edges are drawn in green
    and nodes are drawn in yellow.

    Parameters
    ----------
    image : ndarray
        The image to be drawn on.
    g : RAG
        The Region Adjacency Graph.
    threshold : float
        Only edges in `g` below `threshold` are drawn.

    Returns:
    out: ndarray
        Image with the edges drawn.
    """
    image = image.copy()
    for edge in g.edges():
        n1, n2 = edge

        r1, c1 = map(int, rag.nodes[n1]['centroid'])
        r2, c2 = map(int, rag.nodes[n2]['centroid'])

        # line  = draw.line(r1, c1, r2, c2)
        # circle = draw.circle(r1,c1,2)

        #if g[n1][n2]['weight'] < high and g[n1][n2]['weight'] > low:
        weight_int = g.nodes[n1]['mean color'].astype(int) - g.nodes[n2]['mean color'].astype(int)
        weight_int = np.linalg.norm(weight_int)
        weight_double = g[n1][n2]['weight']

        #print weight_int,weight_double
        if weight_int > 30 and weight_double < 30 :
            print ("Double Vectors")
            print ("Vector 1",g.nodes[n1]['mean color'])
            print ("Vector 2",g.nodes[n2]['mean color'])
            print ("Difference",g.nodes[n1]['mean color'] - g.nodes[n2]['mean color'])
            print ("Magnitude",weight_double)

            print ("Int Vectors")
            print ("Vector 1",g.nodes[n1]['mean color'].astype(int))
            print ("Vector 2",g.nodes[n2]['mean color'].astype(int))
            print ("Difference",g.nodes[n1]['mean color'].astype(int) - g.nodes[n2]['mean color'].astype(int))
            print ("Magnitude",weight_int)

            # image[line] = 0,1,0

        # image[circle] = 1,1,0

    return image

edges_drawn_30 = display_edges(label_rgb, rag)
show_img(edges_drawn_30)
plt.figure(figsize = (12,8))

plt.show()