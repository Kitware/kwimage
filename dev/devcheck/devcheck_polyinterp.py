import numpy as np
from shapely import geometry
import kwplot
import kwimage
plt = kwplot.autoplt()

kwplot.figure(doclf=1)

# shape1 = geometry.box(0, 0, 1, 1).buffer(0)
# shape2 = geometry.Point(3, 0.5).buffer(0.25)

poly1 = kwimage.Polygon.random(7, convex=False)
poly2 = kwimage.Polygon.random(15, convex=False).translate((2, 2))
shape1 = poly1.to_shapely()
shape2 = poly2.to_shapely()


corners1 = np.squeeze(shape1.exterior.xy)[:, :-1]
cannon1 = corners1[:, corners1.sum(axis=0).argmax()]
offset1 = shape1.exterior.project(geometry.Point(cannon1))

corners2 = np.squeeze(shape2.exterior.xy)[:, :-1]
cannon2 = corners2[:, corners2.sum(axis=0).argmax()]
offset2 = shape2.exterior.project(geometry.Point(cannon2))

# num_points = 20
# interps = np.linspace(0, 1, num_points, endpoint=False)

# shape1.exterior.xy

# Find the fractional point along the exterior that each vertex lives on
ring_dist1 = [shape1.exterior.project(geometry.Point(pt), normalized=True)
              for pt in poly1.exterior.data]

# Find the fractional point along the exterior that each vertex lives on
ring_dist2 = [shape2.exterior.project(geometry.Point(pt), normalized=True)
              for pt in poly2.exterior.data]

ring_distB = sorted(set(ring_dist1 + ring_dist2))
interps = np.array(ring_distB)


coords1 = np.array([
    shape1.exterior.interpolate(i).xy
    for i in ((interps * shape1.exterior.length)) % shape1.exterior.length
]).squeeze()
coords2 = np.array([
    shape2.exterior.interpolate(i).xy
    for i in ((interps * shape2.exterior.length)) % shape2.exterior.length
]).squeeze()

interpolated_shapes = [
    geometry.Polygon(coords)
    for coords in np.linspace(coords1, coords2, 5)
]

for shape in interpolated_shapes:
    plt.plot(*shape.exterior.xy)
    pts = kwimage.Points(xy=np.array(shape.exterior.xy).T)
    print(f'pts={pts}')
    zzz = pts.draw(radius=0.01, ec='red', color='green')
    print(f'zzz={zzz}')

poly1.draw(alpha=0.5, vertex=0.02)
poly2.draw(alpha=0.5, vertex=0.02)
