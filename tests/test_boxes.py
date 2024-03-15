import kwimage


def _is_torch_available():
    try:
        import torch  # NOQA
    except ImportError:
        return False
    else:
        return True


def _box_clip_test(boxes):
    """
    Clip test to run over multiple backends.
    """
    x_min, y_min, x_max, y_max = (1, 2, 10, 11)
    # Check implace = False case
    clipped = boxes.clip(x_min, y_min, x_max, y_max, inplace=False)
    assert boxes._impl.any(boxes.data != clipped.data)
    assert clipped._impl.all(clipped.width <= x_max)
    assert clipped._impl.all(clipped.height <= y_max)
    assert clipped._impl.all(clipped.tl_x >= x_min)
    assert clipped._impl.all(clipped.tl_y >= y_min)

    # Check implace = True case
    clipped2 = boxes.clip(1, 2, 10, 11, inplace=True)
    assert clipped2.data is boxes.data
    assert boxes._impl.all(clipped2.data == boxes.data)


def test_boxes_clip_numpy():
    boxes = kwimage.Boxes.random(10, rng=0).toformat('ltrb').scale(20)
    _box_clip_test(boxes)


def test_boxes_clip_torch():
    if not _is_torch_available():
        import pytest
        pytest.skip('no torch')

    boxes = kwimage.Boxes.random(10, rng=0).toformat('ltrb').scale(20)
    boxes = boxes.tensor()
    _box_clip_test(boxes)
