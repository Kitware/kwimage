import numpy as np
import pytest


def _require_cv2():
    cv2 = pytest.importorskip('cv2')
    return cv2


def test_draw_text_non_uint8_opencv5_fallback(monkeypatch):
    """Exercise the OpenCV 5 uint8-only putText compatibility path."""
    cv2 = _require_cv2()
    import kwimage

    real_put_text = cv2.putText

    def uint8_only_put_text(img, *args, **kwargs):
        if np.asarray(img).dtype != np.uint8:
            raise cv2.error(
                "OpenCV(5.0.0): (-215:Assertion failed) "
                "img.depth() == CV_8U in function 'putText'"
            )
        return real_put_text(img, *args, **kwargs)

    monkeypatch.setattr(cv2, 'putText', uint8_only_put_text)

    image = np.zeros((80, 240, 3), dtype=np.float32)
    result = kwimage.draw_text_on_image(
        image,
        'opencv5',
        org=(2, 30),
        color='white',
        lineType=cv2.LINE_AA,
    )

    assert result is image
    assert result.dtype == np.float32
    assert result.max() == 1.0
    # OpenCV 4 disabled antialiasing for non-uint8 images.  The fallback
    # retains that value behavior even though OpenCV 5 renders via uint8.
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_draw_text_two_channel_opencv5_fallback(monkeypatch):
    """Exercise OpenCV 5's rejection of two-channel destinations."""
    cv2 = _require_cv2()
    import kwimage

    real_put_text = cv2.putText

    def restricted_channel_put_text(img, *args, **kwargs):
        image_data = np.asarray(img)
        if image_data.ndim == 3 and image_data.shape[2] not in {1, 3, 4}:
            raise cv2.error(
                "OpenCV(5.0.0): (-215:Assertion failed) "
                "nch == 1 || nch == 3 || nch == 4 in function 'putText'"
            )
        return real_put_text(img, *args, **kwargs)

    monkeypatch.setattr(cv2, 'putText', restricted_channel_put_text)

    image = np.zeros((80, 240, 2), dtype=np.uint8)
    result = kwimage.draw_text_on_image(
        image,
        'opencv5',
        org=(2, 30),
        color='white',
        lineType=cv2.LINE_AA,
    )

    assert result is image
    assert result.shape == (80, 240, 2)
    assert result.dtype == np.uint8
    drawn = np.any(result != 0, axis=2)
    assert np.any(drawn)
    assert np.all(result[drawn] == 255)
    assert np.all(result[~drawn] == 0)


def test_draw_text_masked_array_preserves_mask(monkeypatch):
    cv2 = _require_cv2()
    import kwimage

    real_put_text = cv2.putText

    def uint8_only_put_text(img, *args, **kwargs):
        if np.asarray(img).dtype != np.uint8:
            raise cv2.error('img.depth() == CV_8U')
        return real_put_text(img, *args, **kwargs)

    monkeypatch.setattr(cv2, 'putText', uint8_only_put_text)

    data = np.zeros((80, 240, 3), dtype=np.float32)
    data[0:40] = np.nan
    mask = np.zeros_like(data, dtype=bool)
    mask[10:20, 10:20] = True
    image = np.ma.MaskedArray(data, mask=mask.copy())

    result = kwimage.draw_text_on_image(
        image, 'masked', org=(2, 30), valign='bottom', color='white'
    )

    assert result is image
    assert np.array_equal(result.mask, mask)
    assert np.any(np.isfinite(result.data[0:40]))


def test_mask_get_xywh_accepts_opencv5_find_nonzero_shape(monkeypatch):
    """OpenCV 5 flattens vector<Point> results from (N, 1, 2) to (N, 2)."""
    cv2 = _require_cv2()
    import kwimage

    real_find_nonzero = cv2.findNonZero

    def flattened_find_nonzero(data):
        result = real_find_nonzero(data)
        if result is not None:
            result = result.reshape(-1, 2)
        return result

    monkeypatch.setattr(cv2, 'findNonZero', flattened_find_nonzero)

    data = np.zeros((12, 15), dtype=bool)
    data[3, 4] = True
    data[9, 11] = True
    mask = kwimage.Mask(data, 'c_mask')
    assert mask.get_xywh().tolist() == [4, 3, 7, 6]


def test_mask_contour_vectors_accept_flat_point_arrays(monkeypatch):
    cv2 = _require_cv2()
    import kwimage

    real_find_contours = cv2.findContours
    real_convex_hull = cv2.convexHull

    def flattened_find_contours(*args, **kwargs):
        result = real_find_contours(*args, **kwargs)
        if len(result) == 2:
            contours, hierarchy = result
            contours = [c.reshape(-1, 2) for c in contours]
            if hierarchy is not None:
                hierarchy = hierarchy.reshape(-1, 4)
            return contours, hierarchy
        image, contours, hierarchy = result
        contours = [c.reshape(-1, 2) for c in contours]
        if hierarchy is not None:
            hierarchy = hierarchy.reshape(-1, 4)
        return image, contours, hierarchy

    def flattened_convex_hull(*args, **kwargs):
        return real_convex_hull(*args, **kwargs).reshape(-1, 2)

    monkeypatch.setattr(cv2, 'findContours', flattened_find_contours)
    monkeypatch.setattr(cv2, 'convexHull', flattened_convex_hull)

    data = np.zeros((12, 15), dtype=np.uint8)
    data[2:9, 3:11] = 1
    mask = kwimage.Mask(data, 'c_mask')

    multi_poly = mask.to_multi_polygon()
    hull = mask.get_convex_hull()
    assert len(multi_poly) == 1
    assert hull.ndim == 2
    assert hull.shape[1] == 2
