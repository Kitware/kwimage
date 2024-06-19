

def rasterio_mwe():
    import tempfile
    import kwimage
    from osgeo import osr
    import ubelt as ub

    img_dpath = ub.Path(tempfile.mkdtemp())

    # Create a polygon in "World Space"
    wld_polygon_crs84 = kwimage.Polygon(exterior=[
        [15, -20],
        [14, -30],
        [13, -31],
        [17, -32],
        [18, -28],
        [16, -25],
    ])
    wld_box_crs84 = wld_polygon_crs84.bounding_box()
    wld_box_corners_crs84 = wld_box_crs84.corners()

    # This will correspond to an image we will write.
    img_width, img_height = 200, 300
    image_box = kwimage.Boxes([[0, 0, img_width, img_height]], "xywh")
    image_corners = image_box.corners().astype(float)

    # Using the corners of the image box in pixel space and world box in CRS84,
    # define the affine transform between them.
    tf_img_from_wld = kwimage.Affine.fit(wld_box_corners_crs84, image_corners)

    # Inspecting the transform shows that it is an offset, two positive scale
    # factors, and negligable rotation / skew. This makes sense given the
    # corner points we fit.
    print('wld_box_corners_crs84 =\n{}'.format(ub.urepr(wld_box_corners_crs84, nl=1)))
    print('image_corners =\n{}'.format(ub.urepr(image_corners, nl=1)))
    print('Transform WLD -> IMG ' + ub.urepr(tf_img_from_wld.decompose()))

    # We want to create a random test image with a feature corresponding to the
    # world space we defined. Let's project the world polygon into pixel space
    # using the affine transform we just defined.

    wld_polygon_pxl = wld_polygon_crs84.warp(tf_img_from_wld)

    # Now lets make some random image data and draw our feature on it.
    _imdata = kwimage.grab_test_image("amazon")
    imdata = kwimage.imresize(_imdata, dsize=(img_width, img_height))
    imdata = kwimage.ensure_float01(imdata)
    imdata = wld_polygon_pxl.draw_on(imdata, color='orange', alpha=0.8)

    # We will use kwimage wrappers around gdal to write the data to disk.  This
    # will require passing it the CRS and the transform from pixel space to
    # that CRS.
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs = srs.ExportToWkt()

    tf_wld_from_img = tf_img_from_wld.inv()
    img_fpath = img_dpath / 'mwe-image.tif'
    kwimage.imwrite(
        img_fpath,
        imdata,
        backend="gdal",
        crs=crs,
        transform=tf_wld_from_img,
        overviews="auto",
    )

    ####
    img_fpath2 = img_dpath / 'mwe-image2.tif'
    img_fpath2.delete()
    new_dataset = rasterio.open(
        img_fpath2,
        'w',
        driver='GTiff',
        height=imdata.shape[0],
        width=imdata.shape[1],
        count=1,
        dtype=imdata.dtype,
        crs='+proj=longlat',
        transform=rasterio.Affine(*tf_wld_from_img.matrix[0:2].ravel()),
    )
    new_dataset.write(imdata[:, :, 0], 1)
    new_dataset.close()

    # Now, let's read our image back in and visualize it.

    # First let's draw the image in pixel space and overlay the pixel space
    # world polygon on top of it.
    import kwplot
    kwplot.autompl()
    imdata = kwimage.imread(img_fpath)
    _, ax = kwplot.imshow(imdata, pnum=(1, 3, 1))
    wld_polygon_pxl.draw(edgecolor='black', fill=False)
    ax.set_title('read with gdal')

    # Now let's use rasterio to plot the data.
    ax = kwplot.figure(pnum=(1, 3, 2)).gca()
    source = rasterio.open(img_fpath)
    rasterio.plot.show(source, ax=ax, alpha=0.8)
    wld_polygon_crs84.draw(edgecolor='black', fill=0)
    ax.set_title('written with gdal')

    source2 = rasterio.open(img_fpath2)
    ax = kwplot.figure(pnum=(1, 3, 3)).gca()
    rasterio.plot.show(source2, ax=ax, alpha=0.8)
    ax.set_title('written with rasterio')
    # wld_polygon_crs84.draw(edgecolor='black', fill=0)

    # Hmm, the data is upside down. That seems off.
    from osgeo import gdal
    gdal_info = gdal.Info(str(img_fpath), format='json')
    cornerCoordinates = gdal_info['cornerCoordinates']

    # Gdal reports the upperLeft coordinate as (13, -32).
    # That seems reasonable given that our affine transform was defined to map
    # that point in world space to (0, 0)  in pixel space.
    print('cornerCoordinates = {}'.format(ub.urepr(cornerCoordinates, nl=1)))

    # Hmm, but the rasterio bands are indicating -32 is at the bottom.  that
    # doesn't seem to agree with the metadata in the geotiff. Why is this?
    print(f'source.bounds={source.bounds}')


