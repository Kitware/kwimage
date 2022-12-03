"""
The only home we've ever known at

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
import kwplot
plt = kwplot.autoplt()
url = 'https://upload.wikimedia.org/wikipedia/commons/7/73/Pale_Blue_Dot.png'
fpath = ub.grabdata(url)
im = kwimage.imread(fpath)
kwplot.imshow(im, doclf=True, fnum=0, title='Original Pale Blue Dot')

# Annotate the location of earth in the original image
x1, y1 = 343, 343
w, h = 2, 2
annot_imgspace_box = kwimage.Boxes([[x1, y1, w, h]], 'xywh')
crop_imgspace_box = annot_imgspace_box
sl = annot_imgspace_box.to_slices()[0]


scale_factor = 1
scaled_imgspace_box = annot_imgspace_box.scale(scale_factor, about='center').quantize()
crop_imgspace_box = scaled_imgspace_box
sl = scaled_imgspace_box.to_slices()[0]

crop = im[sl]
crop = kwimage.ensure_float01(crop)
kwplot.imshow(crop, fnum=1, doclf=True)
plt.gca().set_title('Pale Blue Dot')


# Get the relative offset
offset_x = annot_imgspace_box.tl_x.ravel()[0] - crop_imgspace_box.tl_x.ravel()[0]
offset_y = annot_imgspace_box.tl_y.ravel()[0] - crop_imgspace_box.tl_y.ravel()[0]


# Approximate the location of the earth
xy = boxes.xy_center.ravel()
poly = kwimage.Polygon.circle(xy=np.array([0.5 + offset_x, 0.5 + offset_y]), r=0.12)
poly = poly.translate((-0.35, +0.30))
# Handle inconsistant pixel grid definition
c = poly.translate((0.5, -0.5)).centroid
print(poly.centroid)
print(c)
kwplot.imshow(crop, fnum=2, doclf=True)
subpixel_color = kwimage.subpixel_getvalue(crop, np.array([c]))
poly.draw(facecolor=subpixel_color, edgecolor=None)
plt.gca().set_title('Pale Blue Dot with approximate Earth location')
