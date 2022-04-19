

def main():
    import kwimage
    from shapely.affinity import affine_transform

    # Test kwimage versus shapley warp

    kw_poly = kwimage.Polygon.random()
    kw_poly = kwimage.Boxes.random(1).to_polygons()[0]

    sh_poly = kw_poly.to_shapely()
    transform = kwimage.Affine.random()

    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('shapely'):
        with timer:
            # This is faster than fancy indexing
            a, b, x, d, e, y = transform.matrix.ravel()[0:6]
            sh_transform = (a, b, d, e, x, y)
            sh_warp_poly = affine_transform(sh_poly, sh_transform)

    for timer in ti.reset('kwimage'):
        with timer:
            kw_warp_poly = kw_poly.warp(transform)

    kw_warp_poly2 = kwimage.Polygon.from_shapely(sh_warp_poly)
    print(kw_warp_poly2)
    print(kw_warp_poly)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_shapely_vs_kwimage_warp.py
    """
    main()
