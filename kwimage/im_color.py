"""
A class to make it easier to work with single colors.
"""

import numpy as np
import ubelt as ub
from . import im_core
from . import _im_color_data

__all__ = ['Color']

__todo__ = """

    - [ ] This class badly needs a rewrite to conform to the fast init / easy
          coerce constructor paradigm.

    - [ ] Make init faster an coerce more general

    - [ ] The init should take a color tuple / list / array, unmodified
          as well requiring a format argument, which might be inferred by
          coerce.

    - [ ] Should have operations to convert to/from different internal formats.

    - [ ] Keep old auto-coercing init, but have a way to disable it by default.

"""

BASE_COLORS = _im_color_data.BASE_COLORS
TABLEAU_COLORS = _im_color_data.TABLEAU_COLORS
XKCD_COLORS = _im_color_data.XKCD_COLORS
CSS4_COLORS = _im_color_data.CSS4_COLORS
KITWARE_COLORS = _im_color_data.KITWARE_COLORS


def _lookup_colorspace_object(space):
    from colormath import color_objects
    if space == 'rgb':
        cls = color_objects.AdobeRGBColor
    elif space == 'lab':
        cls = color_objects.LabColor
    elif space == 'hsv':
        cls = color_objects.HSVColor
    elif space == 'luv':
        cls = color_objects.LuvColor
    elif space == 'cmyk':
        cls = color_objects.CMYKColor
    elif space == 'cmy':
        cls = color_objects.CMYColor
    elif space == 'xyz':
        cls = color_objects.XYZColor
    else:
        raise KeyError(space)
    return cls


def _colormath_convert(src_color, src_space, dst_space):
    """
    Uses colormath to convert colors

    Example:
        >>> # xdoctest: +REQUIRES(module:colormath)
        >>> import kwimage
        >>> from kwimage.im_color import _colormath_convert
        >>> src_color = kwimage.Color('turquoise').as01()
        >>> print('src_color = {}'.format(ub.urepr(src_color, nl=0, precision=2)))
        >>> src_space = 'rgb'
        >>> dst_space = 'lab'
        >>> lab_color = _colormath_convert(src_color, src_space, dst_space)
        ...
        >>> print('lab_color = {}'.format(ub.urepr(lab_color, nl=0, precision=2)))
        lab_color = (78.11, -70.09, -9.33)
        >>> rgb_color = _colormath_convert(lab_color, 'lab', 'rgb')
        >>> print('rgb_color = {}'.format(ub.urepr(rgb_color, nl=0, precision=2)))
        rgb_color = (0.29, 0.88, 0.81)
        >>> hsv_color = _colormath_convert(lab_color, 'lab', 'hsv')
        >>> print('hsv_color = {}'.format(ub.urepr(hsv_color, nl=0, precision=2)))
        hsv_color = (175.39, 1.00, 0.88)
    """
    from colormath.color_conversions import convert_color
    src_cls = _lookup_colorspace_object(src_space)
    dst_cls = _lookup_colorspace_object(dst_space)
    src = src_cls(*src_color)
    dst = convert_color(src, dst_cls)
    dst_color = dst.get_value_tuple()
    return dst_color


class Color(ub.NiceRepr):
    """
    Used for converting a single color between spaces and encodings.
    This should only be used when handling small numbers of colors(e.g. 1),
    don't use this to represent an image.

    Args:
        space (str): colorspace of wrapped color.
            Assume RGB if not specified and it cannot be inferred

    CommandLine:
        xdoctest -m ~/code/kwimage/kwimage/im_color.py Color

    Example:
        >>> print(Color('g'))
        >>> print(Color('orangered'))
        >>> print(Color('#AAAAAA').as255())
        >>> print(Color([0, 255, 0]))
        >>> print(Color([1, 1, 1.]))
        >>> print(Color([1, 1, 1]))
        >>> print(Color(Color([1, 1, 1])).as255())
        >>> print(Color(Color([1., 0, 1, 0])).ashex())
        >>> print(Color([1, 1, 1], alpha=255))
        >>> print(Color([1, 1, 1], alpha=255, space='lab'))
    """
    def __init__(self, color, alpha=None, space=None, coerce=True):
        """
        Args:
            color (Color | Iterable[int | float] | str):
                something coercable into a color
            alpha (float | None):
                if psecified adds an alpha value
            space (str):
                The colorspace to interpret this color as. Defaults to rgb.
            coerce (bool):
                The exsting init is not lightweight. This is a design problem
                that will need to be fixed in future versions. Setting
                coerce=False will disable all magic and use imputed color and
                space args directly. Alpha will be ignored.
        """
        if coerce:
            try:
                # Hack for ipython reload
                is_color_cls = color.__class__.__name__ == 'Color'
            except Exception:
                is_color_cls = isinstance(color, Color)

            if is_color_cls:
                assert alpha is None
                assert space is None
                space = color.space
                color = color.color01
            else:
                # FIXME: This is a bad check, and it's hard to fix given that there
                # is lots of code that likely depends on this now. We should not be
                # doing coercion in an `__init__`, which should alway be
                # lightweight.  The check_inputs=False can be used to disable this
                # explicitly as a workaround.
                color = self._ensure_color01(color)
                if alpha is not None:
                    alpha = self._ensure_color01([alpha])[0]

            if space is None:
                space = 'rgb'

            # always normalize the color down to 01
            color01 = list(color)

            if alpha is not None:
                if len(color01) not in [1, 3]:
                    raise ValueError('alpha already in color')
                color01 = color01 + [alpha]

            # correct space if alpha is given
            if len(color01) in [2, 4]:
                if not space.endswith('a'):
                    space += 'a'
        else:
            color01 = color
            space = space

        # FIXME: color01 is not a good name because the data wont be between 0
        # and 1 for non-rgb spaces. We should differentiate between rgb01 and
        # rgb255.
        self.color01 = color01
        self.space = space

    @classmethod
    def coerce(cls, data, **kwargs):
        return cls(data, **kwargs)

    def __nice__(self):
        colorpart = ', '.join(['{:.2f}'.format(c) for c in self.color01])
        return self.space + ': ' + colorpart

    def forimage(self, image, space='auto'):
        """
        Return a numeric value for this color that can be used
        in the given image.

        Create a numeric color tuple that agrees with the format of the input
        image (i.e. float or int, with 3 or 4 channels).

        Args:
            image (ndarray): image to return color for
            space (str): colorspace of the input image.
                Defaults to 'auto', which will choose rgb or rgba

        Returns:
            Tuple[Number, ...]: the color value

        Example:
            >>> import kwimage
            >>> img_f3 = np.zeros([8, 8, 3], dtype=np.float32)
            >>> img_u3 = np.zeros([8, 8, 3], dtype=np.uint8)
            >>> img_f4 = np.zeros([8, 8, 4], dtype=np.float32)
            >>> img_u4 = np.zeros([8, 8, 4], dtype=np.uint8)
            >>> kwimage.Color('red').forimage(img_f3)
            (1.0, 0.0, 0.0)
            >>> kwimage.Color('red').forimage(img_f4)
            (1.0, 0.0, 0.0, 1.0)
            >>> kwimage.Color('red').forimage(img_u3)
            (255, 0, 0)
            >>> kwimage.Color('red').forimage(img_u4)
            (255, 0, 0, 255)
            >>> kwimage.Color('red', alpha=0.5).forimage(img_f4)
            (1.0, 0.0, 0.0, 0.5)
            >>> kwimage.Color('red', alpha=0.5).forimage(img_u4)
            (255, 0, 0, 127)
            >>> kwimage.Color('red').forimage(np.uint8)
            (255, 0, 0)
        """
        if space == 'auto':
            space = 'rgb'
        try:
            kind = image.dtype.kind
        except AttributeError:
            kind = np.dtype(image).kind
            if len(self.color01) == 4:
                if not space.endswith('a'):
                    space = space + 'a'
        else:
            if im_core.num_channels(image) == 4:
                if not space.endswith('a'):
                    space = space + 'a'

        if kind == 'f':
            color = self.as01(space)
        else:
            color = self.as255(space)
        return color

    def _forimage(self, image, space='rgb'):
        """ backwards compat, deprecate """
        ub.schedule_deprecation(
            'kwimage', 'Color._forimage', 'method',
            migration='Use forimage instead',
            deprecate='0.10.0', error='1.0.0', remove='1.1.0')
        return self.forimage(image, space)

    def ashex(self, space=None):
        """
        Convert to hex values

        Args:
            space (None | str):
                if specified convert to this colorspace before returning

        Returns:
            str: the hex representation
        """
        c255 = self.as255(space)
        return '#' + ''.join(['{:02x}'.format(c) for c in c255])

    def as255(self, space=None):
        """
        Convert to byte values

        Args:
            space (None | str):
                if specified convert to this colorspace before returning

        Returns:
            Tuple[int, int, int] | Tuple[int, int, int, int]:
                The uint8 tuple of color values between 0 and 255.
        """
        # TODO: be more efficient about not changing to 01 space
        color = tuple(int(c * 255) for c in self.as01(space))
        return color

    def as01(self, space=None):
        """
        Convert to float values

        Args:
            space (None | str):
                if specified convert to this colorspace before returning

        Returns:
            Tuple[float, float, float] | Tuple[float, float, float, float]:
                The float tuple of color values between 0 and 1

        Note:
            This function is only guarenteed to return 0-1 values for rgb
            values. For HSV and LAB, the native spaces are used. This is not
            ideal, and we may create a new function that fixes this - at least
            conceptually - and deprate this for that in the future.

            For HSV, H is between 0 and 360. S, and V are in [0, 1]
        """
        color = tuple(map(float, self.color01))
        if space is not None:
            if space == self.space:
                pass
            elif space == 'rgb' and self.space == 'rgba':
                color = color[0:3]
            elif space == 'rgba' and self.space == 'rgb':
                color = color + (1.0,)
            elif space == 'bgr' and self.space == 'rgb':
                color = color[::-1]
            elif space == 'rgb' and self.space == 'bgr':
                color = color[::-1]
            elif space == 'lab' and self.space == 'rgb':
                # Note: in this case we will not get a 0-1 normalized color.
                # because lab does not natively exist in the 0-1 space.
                color = _colormath_convert(color, 'rgb', 'lab')
            elif space == 'hsv' and self.space == 'rgb':
                color = _colormath_convert(color, 'rgb', 'hsv')
            elif space == 'hsv' and self.space == 'rgba':
                rgba = color
                color = _colormath_convert(rgba[0:3], 'rgb', 'hsv') + rgba[3:4]
            elif space == 'rgb' and self.space == 'hsv':
                color = _colormath_convert(color, 'hsv', 'rgb')
            elif space == 'rgb' and self.space == 'lab':
                color = _colormath_convert(color, 'lab', 'rgb')
            else:
                # from colormath import color_conversions
                raise NotImplementedError('{} -> {}'.format(self.space, space))
        return color

    @classmethod
    def _is_base01(channels):
        """ check if a color is in base 01 """
        def _test_base01(channels):
            tests01 = {
                'is_float': all([isinstance(c, (float, np.float64)) for c in channels]),
                'is_01': all([c >= 0.0 and c <= 1.0 for c in channels]),
            }
            return tests01
        if isinstance(channels, str):
            return False
        return all(_test_base01(channels).values())

    @classmethod
    def _is_base255(Color, channels):
        """ there is a one corner case where all pixels are 1 or less """
        if (all(c > 0.0 and c <= 255.0 for c in channels) and any(c > 1.0 for c in channels)):
            # Definately in 255 space
            return True
        else:
            # might be in 01 or 255
            return all(isinstance(c, int) for c in channels)

    @classmethod
    def _hex_to_01(Color, hex_color):
        """
        hex_color = '#6A5AFFAF'
        """
        assert hex_color.startswith('#'), 'not a hex string %r' % (hex_color,)
        parts = hex_color[1:].strip()
        color255 = tuple(int(parts[i: i + 2], 16) for i in range(0, len(parts), 2))
        assert len(color255) in [3, 4], 'must be length 3 or 4'
        return Color._255_to_01(color255)

    def _ensure_color01(Color, color):
        """ Infer what type color is and normalize to 01 """
        if isinstance(color, str):
            color = Color._string_to_01(color)
        elif Color._is_base255(color):
            color = Color._255_to_01(color)
        return color

    @classmethod
    def _255_to_01(Color, color255):
        """ converts base 255 color to base 01 color """
        return [channel / 255.0 for channel in color255]

    @classmethod
    def _string_to_01(Color, color):
        """
        Ignore:
            mplutil.Color._string_to_01('green')
            mplutil.Color._string_to_01('red')
        """
        if color == 'random':
            import random
            ub.schedule_deprecation(
                'kwimage', 'Color._string_to_01 with random', 'method',
                migration='Use Color.random instead',
                deprecate='0.10.0', error='1.0.0', remove='1.1.0')
            color = random.choice(Color.named_colors())
        if color in BASE_COLORS:
            color01 = BASE_COLORS[color]
        elif color in CSS4_COLORS:
            color_hex = CSS4_COLORS[color]
            color01 = Color._hex_to_01(color_hex)
        elif color in XKCD_COLORS:
            color_hex = XKCD_COLORS[color]
            color01 = Color._hex_to_01(color_hex)
        elif color in KITWARE_COLORS:
            color_hex = KITWARE_COLORS[color]
            color01 = Color._hex_to_01(color_hex)
        elif color.startswith('#'):
            color01 = Color._hex_to_01(color)
        else:
            raise ValueError('unknown color=%r' % (color,))
        return color01

    @classmethod
    def named_colors(cls):
        """
        Returns:
            List[str]: names of colors that Color accepts

        Example:
            >>> import kwimage
            >>> named_colors = kwimage.Color.named_colors()
            >>> color_lut = {name: kwimage.Color(name).as01() for name in named_colors}
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> # This is a very big table if we let it be, reduce it
            >>> color_lut =dict(list(color_lut.items())[0:10])
            >>> canvas = kwplot.make_legend_img(color_lut)
            >>> kwplot.imshow(canvas)
        """
        NAMED_COLORS = set(BASE_COLORS) | set(CSS4_COLORS) | set(XKCD_COLORS) | set(KITWARE_COLORS)
        names = sorted(NAMED_COLORS)
        return names

    @classmethod
    def distinct(Color, num, existing=None, space='rgb', legacy='auto',
                 exclude_black=True, exclude_white=True):
        """
        Make multiple distinct colors.

        The legacy variant is based on a stack overflow post [HowToDistinct]_,
        but the modern variant is based on the :mod:`distinctipy` package.

        References:
            .. [HowToDistinct] https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

            .. [ColorLimits] https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
            .. [WikiDistinguish] https://en.wikipedia.org/wiki/Help:Distinguishable_colors
            .. [Disinct2] https://ux.stackexchange.com/questions/17964/how-many-visually-distinct-colors-can-accurately-be-associated-with-a-separate

        TODO:
            - [ ] If num is more than a threshold we should switch to
               a different strategy to generating colors that just samples
               uniformly from some colormap and then shuffles. We have no hope
               of making things distinguishable when num starts going over
               10 or so. See [ColorLimits]_ [WikiDistinguish]_ [Disinct2]_ for
               more ideas.

        Returns:
            List[Tuple]: list of distinct float color values

        Example:
            >>> # xdoctest: +REQUIRES(module:matplotlib)
            >>> from kwimage.im_color import *  # NOQA
            >>> import kwimage
            >>> colors1 = kwimage.Color.distinct(5, legacy=False)
            >>> colors2 = kwimage.Color.distinct(3, existing=colors1)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--show)
            >>> from kwimage.im_color import _draw_color_swatch
            >>> swatch1 = _draw_color_swatch(colors1, cellshape=9)
            >>> swatch2 = _draw_color_swatch(colors1 + colors2, cellshape=9)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(swatch1, pnum=(1, 2, 1), fnum=1)
            >>> kwplot.imshow(swatch2, pnum=(1, 2, 2), fnum=1)
            >>> kwplot.show_if_requested()
        """
        if legacy == 'auto':
            legacy = (existing is None)

        if legacy:
            import matplotlib as mpl
            import matplotlib._cm  as _cm
            assert existing is None
            # Old behavior
            cm = mpl.colors.LinearSegmentedColormap.from_list(
                'gist_rainbow', _cm.datad['gist_rainbow'],
                mpl.rcParams['image.lut'])

            distinct_colors = [
                np.array(cm(i / num)).tolist()[0:3]
                for i in range(num)
            ]
            if space == 'rgb':
                return distinct_colors
            else:
                return [Color(c, space='rgb').as01(space=space) for c in distinct_colors]
        else:
            from distinctipy import distinctipy
            if space != 'rgb':
                raise NotImplementedError
            exclude_colors = existing
            if exclude_colors is None:
                exclude_colors = []
            if exclude_black:
                exclude_colors = exclude_colors + [(0., 0., 0.)]
            if exclude_white:
                exclude_colors = exclude_colors + [(1., 1., 1.)]
            # convert string to int for seed
            seed = int(ub.hash_data(exclude_colors, base=10)) + num
            distinct_colors = distinctipy.get_colors(
                num, exclude_colors=exclude_colors, rng=seed)
            distinct_colors = [tuple(map(float, c)) for c in distinct_colors]
            return distinct_colors

        if space == 'rgb':
            return distinct_colors
        else:
            return [Color(c, space='rgb').as01(space=space) for c in distinct_colors]

    @classmethod
    def random(Color, pool='named', with_alpha=0, rng=None):
        """
        Returns:
            Color
        """
        import kwarray
        rng = kwarray.ensure_rng(rng, api='python')
        if pool == 'named':
            color_name = rng.choice(Color.named_colors())
            color = Color._string_to_01(color_name)
        else:
            raise NotImplementedError
        if with_alpha:
            color = color + [rng.random()]
        return Color(color)

    def distance(self, other, space='lab'):
        """
        Distance between self an another color

        Args:
            other (Color): the color to compare
            space (str): the colorspace to comapre in

        Returns:
            float

        Ignore:
            import kwimage
            self = kwimage.Color((0.16304347826086973, 0.0, 1.0))
            other = kwimage.Color('purple')

            hard_coded_colors = {
                'a': (1.0, 0.0, 0.16),
                'b': (1.0, 0.918918918918919, 0.0),
                'c': (0.0, 1.0, 0.0),
                'd': (0.0, 0.9239130434782604, 1.0),
                'e': (0.16304347826086973, 0.0, 1.0)
            }

            # Find grays
            names = kwimage.Color.named_colors()
            grays = {}
            for name in names:
                color = kwimage.Color(name)
                r, g, b = color.as01()
                if r == g and g == b:
                    grays[name] = (r, g, b)
            print(ub.urepr(ub.sorted_vals(grays), nl=-1))

            for k, v in hard_coded_colors.items():
                self = kwimage.Color(v)
                distances = []
                for name in names:
                    other = kwimage.Color(name)
                    dist = self.distance(other)
                    distances.append(dist)

                idxs = ub.argsort(distances)[0:5]
                dists = list(ub.take(distances, idxs))
                names = list(ub.take(names, idxs))
                print('k = {!r}'.format(k))
                print('names = {!r}'.format(names))
                print('dists = {!r}'.format(dists))
        """
        vec1 = np.array(self.as01(space))
        vec2 = np.array(other.as01(space))
        return np.linalg.norm(vec1 - vec2)

    def interpolate(self, other, alpha=0.5, ispace=None, ospace=None):
        """
        Interpolate between colors

        Args:
            other (Color): A coercable Color
            alpha (float | List[float]): one or more interpolation values
            ispace (str | None): colorspace to interpolate in
            ospace (str | None): colorspace of returned color

        Returns:
            Color | List[Color]

        Example:
            >>> import kwimage
            >>> color1 = self = kwimage.Color.coerce('orangered')
            >>> color2 = other = kwimage.Color.coerce('dodgerblue')
            >>> alpha = np.linspace(0, 1, 6)
            >>> ispace = 'rgb'
            >>> ospace = 'rgb'
            >>> colorBs = self.interpolate(other, alpha, ispace=ispace, ospace=ospace)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> # xdoctest: +REQUIRES(--show)
            >>> from kwimage.im_color import _draw_color_swatch
            >>> swatch_colors = [color1] + colorBs + [color2]
            >>> print('swatch_colors = {}'.format(ub.urepr(swatch_colors, nl=1)))
            >>> swatch1 = _draw_color_swatch(swatch_colors, cellshape=(8, 8))
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(swatch1, pnum=(1, 1, 1), fnum=1)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        other = kwimage.Color.coerce(other)
        vec1 = np.array(self.as01(ispace))
        vec2 = np.array(other.as01(ispace))
        if ub.iterable(alpha):
            alpha = np.asarray(alpha).ravel()
            vecB = vec1[None, :] * (1 - alpha)[:, None] + (vec2[None, :] * alpha[:, None])
            new = [
                kwimage.Color(
                    kwimage.Color(c, space=ispace, coerce=False).as01(ospace),
                    space=ospace, coerce=False)
                for c in vecB
            ]
        else:
            vecB = vec1 * (1 - alpha) + (vec2 * alpha)
            c = vecB
            new = kwimage.Color(
                kwimage.Color(c, space=ispace, coerce=False).as01(ospace),
                space=ospace, coerce=False)
        return new

    def to_image(self, dsize=(8, 8)):
        """
        Create an solid-color image with this color

        Args:
            dsize (Tuple[int, int]):
                the desired width / height of the image (defaults to 8x8)
        """
        w, h = dsize
        color_arr = np.array(self.as01()).astype(np.float32)
        cell_pixel = color_arr[None, None]
        cell = np.tile(cell_pixel, (h, w, 1))
        return cell

    def adjust(self, saturate=0, lighten=0):
        """
        Adjust the saturation or value of a color.

        Requires that :mod:`colormath` is installed.

        Args:
            saturate (float):
                between +1 and -1, when positive saturates the color, when
                negative desaturates the color.

            lighten (float):
                between +1 and -1, when positive lightens the color, when
                negative darkens the color.

        Example:
            >>> # xdoctest: +REQUIRES(module:colormath)
            >>> import kwimage
            >>> self = kwimage.Color.coerce('salmon')
            >>> new = self.adjust(saturate=+0.2)
            >>> cell1 = self.to_image()
            >>> cell2 = new.to_image()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwimage.stack_images([cell1, cell2], axis=1)
            >>> kwplot.imshow(canvas)

        Example:
            >>> # xdoctest: +REQUIRES(module:colormath)
            >>> import kwimage
            >>> self = kwimage.Color.coerce('salmon', alpha=0.5)
            >>> new = self.adjust(saturate=+0.2)
            >>> cell1 = self.to_image()
            >>> cell2 = new.to_image()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwimage.stack_images([cell1, cell2], axis=1)
            >>> kwplot.imshow(canvas)

        Example:
            >>> # xdoctest: +REQUIRES(module:colormath)
            >>> import kwimage
            >>> adjustments = [
            >>>     {'saturate': -0.2},
            >>>     {'saturate': +0.2},
            >>>     {'lighten': +0.2},
            >>>     {'lighten': -0.2},
            >>>     {'saturate': -0.9},
            >>>     {'saturate': +0.9},
            >>>     {'lighten': +0.9},
            >>>     {'lighten': -0.9},
            >>> ]
            >>> self = kwimage.Color.coerce('kitware_green')
            >>> dsize = (256, 64)
            >>> to_show = []
            >>> to_show.append(self.to_image(dsize))
            >>> for kwargs in adjustments:
            >>>     new = self.adjust(**kwargs)
            >>>     cell = new.to_image(dsize=dsize)
            >>>     text = ub.urepr(kwargs, compact=1, nobr=1)
            >>>     cell, info = kwimage.draw_text_on_image(cell, text, return_info=1, border={'thickness': 2}, color='white', fontScale=1.0)
            >>>     to_show.append(cell)
            >>> # xdoctest: +REQUIRES(--show)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwimage.stack_images_grid(to_show)
            >>> canvas = kwimage.draw_header_text(canvas, 'kwimage.Color.adjust')
            >>> kwplot.imshow(canvas)
        """
        h, s, v, *a = self.as01(space='hsv')
        assert 0 <= h <= 360
        assert 0 <= s <= 1
        assert 0 <= v <= 1
        if saturate:
            s = max(min(1, s + saturate), 0)
        if lighten:
            v = max(min(1, v + lighten), 0)
        hsv = (h, s, v)
        rgb = _colormath_convert(hsv, 'hsv', 'rgb')
        color = list(rgb) + a
        new_rgb = self.__class__.coerce(color)
        return new_rgb


def _draw_color_swatch(colors, cellshape=9):
    """
    Draw colors in a grid

    Ignore:
        # https://seaborn.pydata.org/tutorial/color_palettes.html

        import kwplot
        sns = kwplot.sns
        from kwimage.im_color import *  # NOQA
        from kwimage.im_color import _draw_color_swatch
        colors = sns.palettes.color_palette('deep', n_colors=10)
        swatch = _draw_color_swatch(colors)
        kwplot.imshow(swatch)
    """
    import kwimage
    import math
    if not ub.iterable(cellshape):
        cellshape = [cellshape, cellshape]
    cell_h = cellshape[0]
    cell_w = cellshape[1]
    cells = []
    for color in colors:
        cell = kwimage.Color(color).to_image(dsize=(cell_w, cell_h))
        cells.append(cell)

    num_colors = len(colors)
    num_cells_side0 = max(1, int(np.sqrt(num_colors)))
    num_cells_side1 = math.ceil(num_colors / num_cells_side0)
    num_cells = num_cells_side1 * num_cells_side0
    num_null_cells = num_cells - num_colors
    if num_null_cells > 0:
        null_cell = np.zeros((cell_h, cell_w, 3), dtype=np.float32)
        pts1 = np.array([(0, 0),                   (cell_w - 1, 0)])
        pts2 = np.array([(cell_w - 1, cell_h - 1), (0, cell_h - 1)])
        null_cell = kwimage.draw_line_segments_on_image(
            null_cell, pts1, pts2, color='red')
        # null_cell = kwimage.draw_text_on_image(
        #     {'width': cell_w, 'height': cell_h}, text='X', color='red',
        #     halign='center', valign='center')
        null_cell = kwimage.ensure_float01(null_cell)
        cells.extend([null_cell] * num_null_cells)
    swatch = kwimage.stack_images_grid(
        cells, chunksize=num_cells_side0, axis=0)
    return swatch
