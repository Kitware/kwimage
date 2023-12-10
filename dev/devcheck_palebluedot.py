"""
The Pale Blue Dot. The only home we've ever known.

Notes:

    np.array([[[100,  82, 106], [100, 119, 104]],
              [[145, 121, 173], [149, 179, 158]]], dtype=uint8)

    It occupies ~0.12 subpixels in this image.

    Earth diameter is 12.742 million meters.

    w_earth_pixel = 0.12  # pixels
    w_earth_meter = 12.742e6  # meters
    106 million GSD

This 4 pixel image has a resolution of 106 million GSD.  You may ask: how can a
GSD be 106 million when the earth diameter is only ~12.7 million meters? The
answer is that Earth is only ~0.12 pixels wide somewhere in this image, but it
likely is somewhere in the lower center left. Not sure if it's possible to
calculate a spread on its position.

References:
    [SE2867] https://skeptics.stackexchange.com/questions/2867
"""

import kwimage
import ubelt as ub
import numpy as np
import kwplot


def main():
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/73/Pale_Blue_Dot.png'
    fpath = ub.grabdata(url)
    im = kwimage.imread(fpath)

    # Annotate the location of earth in the original image
    x1, y1 = 343, 343
    w, h = 2, 2
    annot_imgspace_box = kwimage.Boxes([[x1, y1, w, h]], 'xywh')
    sl = annot_imgspace_box.to_slices()[0]

    scale_factor = 4
    scaled_imgspace_box = annot_imgspace_box.scale(scale_factor, about='center').quantize()
    crop_imgspace_box = scaled_imgspace_box
    sl = scaled_imgspace_box.to_slices()[0]

    crop = im[sl]
    crop = kwimage.ensure_float01(crop)

    # Get the relative offset
    offset_x = annot_imgspace_box.tl_x.ravel()[0] - crop_imgspace_box.tl_x.ravel()[0]
    offset_y = annot_imgspace_box.tl_y.ravel()[0] - crop_imgspace_box.tl_y.ravel()[0]

    # Approximate the location of the earth
    earth_polygon = kwimage.Polygon.circle(xy=np.array([0.5 + offset_x, 0.5 + offset_y]), r=0.12)
    earth_polygon = earth_polygon.translate((-0.35, +0.30))
    # Handle inconsistant pixel grid definition
    earth_centroid = earth_polygon.translate((0.5, -0.5)).centroid
    print(earth_centroid)

    # Given the Earth's estimated location, determine the pixel of that subpixel
    subpixel_color = kwimage.subpixel_getvalue(crop, np.array([earth_centroid]))

    # Draw Plots
    plt = kwplot.autoplt()
    kwplot.imshow(im, doclf=True, fnum=1, title='Pale Blue Dot\n(Image from WikiMedia)', pnum=(2, 2, 1))
    # scaled_imgspace_box.translate((-0.5, -0.5)).draw()

    plt = kwplot.autoplt()
    kwplot.imshow(im, fnum=1, title='Pale Blue Dot\n(Annotated)', pnum=(2, 2, 3))
    scaled_imgspace_box.translate((-0.5, -0.5)).draw()

    kwplot.imshow(crop, fnum=1, pnum=(2, 2, 2))
    plt.gca().set_title('Pale Blue Dot\n(zoomed in)')

    kwplot.imshow(crop, fnum=1, pnum=(2, 2, 4))
    earth_polygon.draw(facecolor=subpixel_color, edgecolor=(0.3, 0.3, 0.3))
    plt.gca().set_title('Pale Blue Dot\n(with approximate Earth location)')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/devcheck_palebluedot.py
    """
    main()
