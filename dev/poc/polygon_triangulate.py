"""
TODO: we want a feature that lets us triangulate a 2D polygon so we can render
it in 3D.
"""


def triangulate_polygon_interior(polygon):
    """
    References:
        https://gis.stackexchange.com/questions/316697/delaunay-triangulation-algorithm-in-shapely-producing-erratic-result
    """
    import numpy as np
    # from shapely.geometry import Polygon
    # import shapely.wkt
    # from shapely.ops import triangulate
    import geopandas as gpd
    from geovoronoi import voronoi_regions_from_coords
    from scipy.spatial import Delaunay
    import kwimage

    poly_points = []
    gdf_poly_exterior = gpd.GeoDataFrame({'geometry': [polygon.buffer(-0.0000001).exterior]}).explode(index_parts=True).reset_index()
    for geom in gdf_poly_exterior.geometry:
        poly_points += np.array(geom.coords).tolist()

    try:
        polygon.interiors[0]
    except Exception:
        poly_points = poly_points
    else:
        gdf_poly_interior = gpd.GeoDataFrame({'geometry': [polygon.interiors]}).explode(index_parts=True).reset_index()
        for geom in gdf_poly_interior.geometry:
            poly_points += np.array(geom.coords).tolist()

    poly_points = np.array([item for sublist in poly_points for item in sublist]).reshape(-1, 2)

    poly_shapes, pts = voronoi_regions_from_coords(poly_points, polygon)
    gdf_poly_voronoi = gpd.GeoDataFrame({'geometry': poly_shapes}).explode(index_parts=True).reset_index()

    final_points_accum = []
    final_simplices_accum = []
    index_offset = 0
    # tri_geom = []
    for geom in gdf_poly_voronoi.geometry:
        geom_exterior = np.vstack(geom.exterior.xy).T
        tri = Delaunay(geom_exterior)

        inside_row_indexes = []
        for row_index, simplex_idxs in enumerate(tri.simplices):
            centroid = kwimage.Polygon(exterior=tri.points[simplex_idxs]).to_shapely().centroid
            if centroid.within(polygon):
                inside_row_indexes.append(row_index)

        if len(inside_row_indexes):
            inside_simplicies = tri.simplices[inside_row_indexes]
            final_points_accum.append(tri.points)
            final_simplices_accum.append(inside_simplicies + index_offset)
            index_offset += len(geom_exterior)

        # inside_triangles = [tri for tri in triangulate(geom) if tri.centroid.within(polygon)]
        # tri_geom += inside_triangles
    final_points = np.concatenate(final_points_accum, axis=0)
    final_simplicies = np.concatenate(final_simplices_accum, axis=0)
    return final_points, final_simplicies

    # gdf_poly_triangles = gpd.GeoDataFrame({'geometry': tri_geom})

    # if 0:
    #     import kwplot
    #     kwplot.autompl()
    #     gdf_poly_voronoi.plot()

    #     gdf_poly_exterior.plot()
    #     if 'gdf_poly_interior' in locals():
    #         gdf_poly_interior.plot()
    #     gdf_poly_triangles.plot()
