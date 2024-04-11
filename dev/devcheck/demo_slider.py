import kwarray
import tqdm
import numpy as np


def demo_with_heatmap():
    window_shape = (512, 512)
    big_image = (np.random.rand(2048, 2048, 3).astype(np.float32) * 512).round()

    def process_func(data):
        """
        Example process func
        """
        odd_flags = data % 2 == 1
        new_data = data.copy()
        new_data[odd_flags] = 3 * data[odd_flags] + 1
        new_data[~odd_flags] = data[~odd_flags] / 2
        new_data = new_data.sum(axis=2, keepdims=True)
        return new_data

    slider = kwarray.SlidingWindow(big_image.shape[0:2], window_shape,
                                   overlap=0.3, keepbound=True,
                                   allow_overshoot=True)

    out_channels = 1
    output_shape = slider.input_shape + (out_channels,)

    stitcher = kwarray.Stitcher(output_shape)

    for sl in tqdm.tqdm(slider, desc='sliding window'):
        chip = big_image[sl]
        new_chip = process_func(chip)

        # Basic add
        # stitcher.add(sl, new_chip)

        # Special weighted add
        _stitcher_center_weighted_add(stitcher, sl, new_chip)

    final = stitcher.finalize()
    print('final = {!r}'.format(final))


def _stitcher_center_weighted_add(stitcher, space_slice, data):
    """
    special adding function that downweights edges
    """
    import kwimage
    weights = kwimage.gaussian_patch(data.shape[0:2])[..., None]
    if stitcher.shape[0] < space_slice[0].stop or stitcher.shape[1] < space_slice[1].stop:
        # By embedding the space slice in the stitcher dimensions we can get a
        # slice corresponding to the valid region in the stitcher, and the extra
        # padding encode the valid region of the data we are trying to stitch into.
        subslice, padding = kwarray.embed_slice(space_slice[0:2], stitcher.shape)
        output_slice = (
            slice(padding[0][0], data.shape[0] - padding[0][1]),
            slice(padding[1][0], data.shape[1] - padding[1][1]),
        )
        subdata = data[output_slice]
        subweights = weights[output_slice]

        stitch_slice = subslice
        stitch_data = subdata
        stitch_weights = subweights
    else:
        # Normal case
        stitch_slice = space_slice
        stitch_data = data
        stitch_weights = weights

    # Handle stitching nan values
    invalid_output_mask = np.isnan(stitch_data)
    if np.any(invalid_output_mask):
        spatial_valid_mask = (1 - invalid_output_mask.any(axis=2, keepdims=True))
        stitch_weights = stitch_weights * spatial_valid_mask
        stitch_data[invalid_output_mask] = 0
    stitcher.add(stitch_slice, stitch_data, weight=stitch_weights)


def demo_with_boxes():
    import kwimage
    import kwarray

    rng = kwarray.ensure_rng(0)

    # A dummy big image
    big_image = rng.rand(2048, 2048, 3)

    def detector(data):
        """
        A dummy detector. Plugin whatever you want here.
        """
        h, w = data.shape[0:2]
        n = rng.randint(0, 4)
        dets = kwimage.Detections(
            boxes=kwimage.Boxes.random(n),
            scores=rng.rand(n),
        ).scale((w, h))
        return dets

    # The slider generates slices that index into the window according to a
    # requested scheme.
    window_shape = (512, 512)
    slider = kwarray.SlidingWindow(big_image.shape[0:2], window_shape,
                                   overlap=0.3, keepbound=True,
                                   allow_overshoot=True)

    det_accum = []
    for slices in slider:

        data = big_image[slices]

        # Get detections relative to the window
        rel_dets = detector(data)

        # Put them into absolute coordinates
        offset_x = slices[0].start
        offset_y = slices[1].start
        abs_dets = rel_dets.translate((offset_x, offset_y))

        det_accum.append(abs_dets)

    # Merge all the boxes together
    all_boxes = kwimage.Detections.concatenate(det_accum)

    keep_idxs = all_boxes.non_max_supression()
    final_boxes = all_boxes.take(keep_idxs)
    return final_boxes


if __name__ == '__main__':
    demo_with_heatmap()
