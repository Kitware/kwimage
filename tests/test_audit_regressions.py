import numpy as np
import pytest


def test_boolean_environ_false_values(monkeypatch):
    from kwimage._internal import _boolean_environ

    for value in ['false', 'off', 'no', '0']:
        monkeypatch.setenv('KWIMAGE_TEST_BOOLEAN', value)
        assert _boolean_environ('KWIMAGE_TEST_BOOLEAN', default=True) is False

    for value in ['true', 'on', 'yes', '1']:
        monkeypatch.setenv('KWIMAGE_TEST_BOOLEAN', value)
        assert _boolean_environ('KWIMAGE_TEST_BOOLEAN', default=False) is True

    monkeypatch.delenv('KWIMAGE_TEST_BOOLEAN', raising=False)
    assert _boolean_environ('KWIMAGE_TEST_BOOLEAN', default=True) is True


def test_imcrop_linear_uses_xy_center_order():
    import kwimage

    image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    linear = kwimage.imcrop(
        image, dsize=(3, 3), about=(5, 8), interpolation='linear'
    )
    nearest = kwimage.imcrop(
        image, dsize=(3, 3), about=(5, 8), interpolation='nearest'
    )
    assert np.array_equal(linear, nearest)
    assert linear[1, 1] == image[8, 5]


def test_imcrop_accepts_one_automatic_dimension():
    import kwimage

    image = np.zeros((10, 20), dtype=np.uint8)
    assert kwimage.imcrop(image, dsize=(5, None)).shape == (2, 5)
    assert kwimage.imcrop(image, dsize=(None, 5)).shape == (5, 10)


def test_string_border_value_coerces_border_mode():
    import cv2
    from kwimage.im_cv2 import _coerce_border_mode_value

    image = np.zeros((2, 3), dtype=np.uint8)
    border_mode, border_value = _coerce_border_mode_value(
        None, 'replicate', image
    )
    assert border_mode == cv2.BORDER_REPLICATE
    assert border_value == (0,)


def test_single_pixel_resize_preserves_dsize_order():
    import kwimage

    gray = np.ones((1, 1), dtype=np.uint8)
    color = np.ones((1, 1, 3), dtype=np.uint8)
    assert kwimage.imresize(gray, dsize=(3, 2)).shape == (2, 3)
    assert kwimage.imresize(color, dsize=(3, 2)).shape == (2, 3, 3)


def test_letterbox_forwards_border_value():
    import kwimage

    image = np.ones((2, 4), dtype=np.uint8)
    result = kwimage.imresize(
        image, dsize=(6, 6), letterbox=True, border_value=37
    )
    assert result[0, 0] == 37
    assert result[-1, -1] == 37


def test_morphology_kernel_uses_width_height_order():
    from kwimage.im_cv2 import _morph_kernel_core

    kernel = _morph_kernel_core(3, 7, 'rect')
    assert kernel.shape == (7, 3)


def test_connected_components_accepts_uint16_dtype_forms():
    import kwimage

    image = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    for ltype in ['uint16', np.uint16, np.dtype('uint16'), np.int16]:
        labels, info = kwimage.connected_components(
            image, ltype=ltype, with_stats=False
        )
        assert labels.dtype == np.uint16
        assert info['num_labels'] == 2


def test_make_channels_comparable_atleast3d():
    import kwimage

    for shape2 in [(3, 4), (1, 4)]:
        image1 = np.zeros((3, 4))
        image2 = np.zeros(shape2)
        got1, got2 = kwimage.make_channels_comparable(
            image1, image2, atleast3d=True
        )
        assert got1.shape == (3, 4, 1)
        assert got2.shape == shape2 + (1,)


def test_pixel_distance_does_not_underflow_unsigned_data():
    from kwimage.im_core import _get_pixel_dist

    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[1, 1] = 0
    pixel = np.array([255, 255, 255], dtype=np.uint8)
    distance = _get_pixel_dist(image, pixel)
    assert distance[0, 0] == 0
    assert distance[1, 1] == 255 * 3


def test_matrix_imatmul_updates_and_returns_self():
    import kwimage

    lhs = np.array([[1.0, 2.0], [3.0, 4.0]])
    rhs = np.array([[0.0, 1.0], [1.0, 0.0]])
    matrix = kwimage.Matrix(lhs.copy())
    original = matrix
    matrix @= rhs
    assert matrix is original
    assert np.array_equal(matrix.matrix, lhs @ rhs)


def test_affine_concise_preserves_anisotropic_scale():
    import kwimage

    transform = kwimage.Affine.affine(scale=(2.0, 1.0))
    concise = transform.concise()
    assert 'scale' in concise
    reconstructed = kwimage.Affine.coerce(concise)
    assert np.allclose(reconstructed.matrix, transform.matrix)


def test_projective_identity_is_3x3():
    import kwimage

    assert np.asarray(kwimage.Projective(None)).shape == (3, 3)
    assert np.asarray(kwimage.Projective.eye()).shape == (3, 3)


def test_python_rle_decoder_keeps_final_count():
    from kwimage.im_runlen import _rle_bytes_to_array

    assert _rle_bytes_to_array(b'11', impl='python').tolist() == [1, 1]


def test_numexpr1_alpha_blend_returns_result():
    pytest.importorskip('numexpr')
    from kwimage.im_alphablend import (
        _alpha_blend_numexpr1,
        _alpha_blend_numexpr2,
    )

    rgb1 = np.zeros((2, 3, 3), dtype=np.float32)
    rgb2 = np.ones((2, 3, 3), dtype=np.float32)
    alpha1 = np.full((2, 3), 0.25, dtype=np.float32)
    alpha2 = np.full((2, 3), 0.75, dtype=np.float32)
    got_rgb, got_alpha = _alpha_blend_numexpr1(
        rgb1, alpha1, rgb2, alpha2
    )
    want_rgb, want_alpha = _alpha_blend_numexpr2(
        rgb1, alpha1, rgb2, alpha2
    )
    assert np.allclose(got_rgb, want_rgb)
    assert np.allclose(got_alpha, want_alpha)


def test_color_is_base01_classmethod():
    import kwimage

    assert kwimage.Color._is_base01([0.0, 0.5, 1.0])
    assert not kwimage.Color._is_base01([0, 128, 255])


def test_make_orimask_default_magnitude():
    pytest.importorskip('matplotlib')
    import kwimage

    radians = np.zeros((2, 3), dtype=np.float32)
    result = kwimage.make_orimask(radians, mag=None, alpha=0.5)
    assert result.shape == (2, 3, 4)
    assert np.all(result[..., 3] == 0.5)


def test_polygon_fill_more_than_four_channels():
    import kwimage

    polygon = kwimage.Polygon(exterior=np.array([[1, 1], [4, 1], [4, 4], [1, 4], [1, 1]]))
    image = np.zeros((6, 6, 5), dtype=np.uint8)
    result = polygon.fill(image, value=(1, 2, 3, 4, 5))
    assert result[2, 2].tolist() == [1, 2, 3, 4, 5]


def test_regular_polygon_has_requested_number_of_sides():
    import kwimage

    polygon = kwimage.Polygon.regular(5)
    # The exterior explicitly repeats the first point to close the ring.
    assert len(polygon.exterior.data) == 6
    assert np.allclose(polygon.exterior.data[0], polygon.exterior.data[-1])


def test_polygon_clockwise_includes_closing_edge():
    from kwimage.structs.polygon import _is_clockwise

    # This ordering is clockwise, but omitting the final edge back to the
    # first vertex produces the opposite answer.
    vertices = np.array([[0, 0], [0, 2], [1, 1]], dtype=float)
    assert _is_clockwise(vertices)
    assert not _is_clockwise(vertices[::-1])


def test_empty_object_list_concatenate():
    import kwimage

    result = kwimage.MaskList.concatenate([])
    assert isinstance(result, kwimage.MaskList)
    assert len(result) == 0


def test_points_round_non_inplace_does_not_mutate_original():
    import kwimage

    original = kwimage.Points(xy=np.array([[0.2, 1.8]]))
    rounded = original.round(inplace=False)
    assert np.allclose(original.xy, [[0.2, 1.8]])
    assert np.allclose(rounded.xy, [[0.0, 2.0]])
    assert rounded.data is not original.data


def test_integer_box_intersection_handles_disjoint_boxes():
    import kwimage

    boxes1 = kwimage.Boxes(np.array([[0, 0, 1, 1]]), 'ltrb')
    boxes2 = kwimage.Boxes(np.array([[2, 2, 3, 3]]), 'ltrb')
    result = boxes1.intersection(boxes2)
    assert result.data.dtype.kind == 'f'
    assert np.isnan(result.data).all()


def test_box_isect_area_forwards_bias():
    import kwimage

    boxes1 = kwimage.Boxes(np.array([[0, 0, 0, 0]]), 'ltrb')
    boxes2 = kwimage.Boxes(np.array([[0, 0, 0, 0]]), 'ltrb')
    assert boxes1.isect_area(boxes2, bias=0)[0, 0] == 0
    assert boxes1.isect_area(boxes2, bias=1)[0, 0] == 1


def test_coco_rle_without_shape_uses_supplied_dims():
    from kwimage.structs.segmentation import _coerce_coco_segmentation

    data = {'counts': [1, 1]}
    result = _coerce_coco_segmentation(data, dims=(1, 2))
    assert data['shape'] == (1, 2)
    assert result.shape == (1, 2)


def test_mask_from_text_height_padding_preserves_rows():
    import kwimage

    mask = kwimage.Mask.from_text('o', shape=(2, 1))
    assert mask.data.tolist() == [[1], [0]]


def test_detections_compress_moves_flags_to_data_device(monkeypatch):
    torch = pytest.importorskip('torch')
    import kwimage
    from kwimage.structs import _generic

    boxes = kwimage.Boxes(
        torch.empty((3, 4), dtype=torch.float32, device='meta'), 'ltrb'
    )
    detections = kwimage.Detections(
        boxes=boxes,
        scores=torch.empty(3, dtype=torch.float32, device='meta'),
    )
    flags = torch.tensor([True, False, True], device='cpu')

    seen_devices = []

    def capture_flags(data, flags, axis=0):
        seen_devices.append(flags.device)
        return data

    monkeypatch.setattr(_generic, '_safe_compress', capture_flags)
    detections.compress(flags)
    assert seen_devices
    assert all(device == detections.device for device in seen_devices)


def test_dense_detection_targets_order_by_box_area():
    import kwimage
    from kwimage.structs.detections import _dets_to_fcmaps

    # The tall box has the larger true area but the smaller squared width.
    # Larger objects are rasterized first, so the smaller square must own the
    # overlap regardless of aspect ratio.
    boxes = kwimage.Boxes(
        np.array([
            [20, 5, 2, 50],
            [18, 25, 6, 6],
        ], dtype=float),
        'xywh',
    )
    detections = kwimage.Detections(
        boxes=boxes,
        class_idxs=np.array([1, 2]),
        classes=['background', 'tall', 'square'],
    )
    target = _dets_to_fcmaps(
        detections,
        bg_size=(0, 0),
        input_dims=(60, 50),
        bg_idx=0,
        soft=False,
        exclude=['diameter', 'offset'],
    )
    assert target['cidx'][28, 21] == 2
