#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CropBorderCLI(scfg.DataConfig):
    """
    Crop uniform borders from an image.

    This script reads an input image, detects and removes uniform-colored (e.g.
    all white) borders, and then saves the result to the specified destination.
    If no destination is provided, the original image is overwritten.
    """
    __command__ = 'crop_border'
    src = scfg.Value(None, position=1, help="Path to the input image.")
    dst = scfg.Value(None, position=2, help="Path to save the cropped image. Defaults to overwriting the input file if not specified.")

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from kwimage.cli.crop_border import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = CropBorderCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        import kwimage
        src_fpath = ub.Path(config.src)
        if config.dst is None:
            dst_fpath = src_fpath
        else:
            dst_fpath = ub.Path(config.dst)
        imdata = kwimage.imread(src_fpath)
        imdata = crop_border_by_color(imdata)
        kwimage.imwrite(dst_fpath, imdata)


def crop_border_by_color(img, fillval=None, thresh=0, channel=None):
    r"""
    Crops image to remove any constant color padding.

    Note: ported from kwplot.mpl_make, needs to move to kwimage proper.

    Args:
        img (NDArray):
            image data

        fillval (None):
            The color to replace.
            Defaults "white" (i.e. `(255,) * num_channels`)

        thresh (int):
            Allowable difference to `fillval` (default = 0)

    Returns:
        ndarray: cropped_img

    TODO:
        does this belong in kwimage?
    """
    import kwimage
    import numpy as np
    if fillval is None:
        fillval = np.array([255] * kwimage.num_channels(img))
    # for colored images
    #with ut.embed_on_exception_context:
    pixel = fillval
    dist = get_pixel_dist(img, pixel, channel=channel)
    isfill = dist <= thresh
    # isfill should just be 2D
    # Fix shape that comes back as (1, W, H)
    if len(isfill.shape) == 3 and isfill.shape[0] == 1:
        if np.all(np.greater(isfill.shape[1:2], [4, 4])):
            isfill = isfill[0]
    rowslice, colslice = _get_crop_slices(isfill)
    cropped_img = img[rowslice, colslice]
    return cropped_img


def get_pixel_dist(img, pixel, channel=None):
    """
    Note: ported from kwplot.mpl_make, needs to move to kwimage proper.

    Example:
        >>> import numpy as np
        >>> img = np.random.rand(256, 256, 3)
        >>> pixel = np.random.rand(3)
        >>> channel = None
        >>> get_pixel_dist(img, pixel, channel)
    """
    import kwimage
    import numpy as np
    pixel = np.asarray(pixel)
    if len(pixel.shape) < 2:
        pixel = pixel[None, None, :]
    img, pixel = kwimage.make_channels_comparable(img, pixel)
    dist = np.abs(img - pixel)
    if len(img.shape) > 2:
        if channel is None:
            dist = np.sum(dist, axis=2)
        else:
            dist = dist[:, :, channel]
    return dist


def _get_crop_slices(isfill):
    """
    Note: ported from kwplot.mpl_make, needs to move to kwimage proper.
    """
    import numpy as np
    import kwarray
    fill_colxs = [np.where(row)[0] for row in isfill]
    fill_rowxs = [np.where(col)[0] for col in isfill.T]
    nRows, nCols = isfill.shape[0:2]
    from functools import reduce
    filled_columns = reduce(np.intersect1d, fill_colxs)
    filled_rows = reduce(np.intersect1d, fill_rowxs)

    consec_rows_list = kwarray.group_consecutive(filled_rows)
    consec_cols_list = kwarray.group_consecutive(filled_columns)

    def get_consec_endpoint(consec_index_list, endpoint):
        """
        consec_index_list = consec_cols_list
        endpoint = 0
        """
        for consec_index in consec_index_list:
            if np.any(np.array(consec_index) == endpoint):
                return consec_index

    def get_min_consec_endpoint(consec_rows_list, endpoint):
        consec_index = get_consec_endpoint(consec_rows_list, endpoint)
        if consec_index is None:
            return endpoint
        return max(consec_index)

    def get_max_consec_endpoint(consec_rows_list, endpoint):
        consec_index = get_consec_endpoint(consec_rows_list, endpoint)
        if consec_index is None:
            return endpoint + 1
        return min(consec_index)

    consec_rows_top    = get_min_consec_endpoint(consec_rows_list, 0)
    consec_rows_bottom = get_max_consec_endpoint(consec_rows_list, nRows - 1)
    remove_cols_left   = get_min_consec_endpoint(consec_cols_list, 0)
    remove_cols_right  = get_max_consec_endpoint(consec_cols_list, nCols - 1)
    rowslice = slice(consec_rows_top, consec_rows_bottom)
    colslice = slice(remove_cols_left, remove_cols_right)
    return rowslice, colslice


__cli__ = CropBorderCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/kwimage/kwimage/cli/crop_border.py
        python -m kwimage.cli.crop_border
    """
    __cli__.main()
