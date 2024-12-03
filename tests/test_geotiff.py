def test_geotiff():
    """
    """
    from kwimage import im_io
    if not im_io._have_gdal():
        import pytest
        pytest.skip()

    import kwimage
    import ubelt as ub
    dpath = ub.Path.appdir('kwimage/tests/geotiff').ensuredir()
    fpath = dpath / 'dummy_geotiff.tif'

    # Make a random polygon in world space
    lat_max = 180 - 1e-7
    lon_max = 90 - 1e-7
    wld_poly = kwimage.Polygon.random().translate((-.5, -.5)).scale(2).scale((lat_max, lon_max))

    # With bounds that are associated to an image of size
    img_dsize = (640, 480)
    # Make a dummy geotiff
    _imdata = kwimage.grab_test_image("amazon")
    imdata = kwimage.imresize(_imdata, dsize=img_dsize)
    imdata = kwimage.ensure_float01(imdata)

    # Given this information, construct the geotiff metadata
    wld_box = wld_poly.box()
    img_box = kwimage.Box.from_dsize(img_dsize)
    wld_corners = wld_box.corners()
    img_corners = img_box.corners().astype(float)

    # The geotransform
    tf_wld_from_img = kwimage.Affine.fit(img_corners, wld_corners)

    # The CRS should be CRS-84
    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs = srs.ExportToWkt()

    nodata_value = 0
    imdata = kwimage.ensure_uint255(imdata)

    # Add in some nodata_value
    imdata[-10:, 10:] = nodata_value
    imdata[0:10:, -200:-180] = nodata_value
    imdata = imdata[:, :, 0:3]

    metadata = {
        'SENSOR': 'kwimage-demo',
    }

    kwimage.imwrite(
        fpath,
        imdata,
        backend="gdal",
        nodata_value=nodata_value,
        crs=crs,
        transform=tf_wld_from_img,
        overviews="auto",
        metadata=metadata,
    )

    from osgeo import gdal
    import os
    dset = gdal.Open(os.fspath(fpath), gdal.GA_ReadOnly)
    info = gdal.Info(dset, format='json', allMetadata=True, listMDD=True)
    print('info = {}'.format(ub.urepr(info, nl=True)))
