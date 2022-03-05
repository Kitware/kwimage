import numpy as np
import ubelt as ub
from collections import OrderedDict
from . import im_core


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
        >>> src_color = kwimage.Color('turquoise').as01()
        >>> print('src_color = {}'.format(ub.repr2(src_color, nl=0, precision=2)))
        >>> src_space = 'rgb'
        >>> dst_space = 'lab'
        >>> lab_color = _colormath_convert(src_color, src_space, dst_space)
        >>> print('lab_color = {}'.format(ub.repr2(lab_color, nl=0, precision=2)))
        lab_color = (78.11, -70.09, -9.33)
        >>> rgb_color = _colormath_convert(lab_color, 'lab', 'rgb')
        >>> print('rgb_color = {}'.format(ub.repr2(rgb_color, nl=0, precision=2)))
        rgb_color = (0.29, 0.88, 0.81)
        >>> hsv_color = _colormath_convert(lab_color, 'lab', 'hsv')
        >>> print('hsv_color = {}'.format(ub.repr2(hsv_color, nl=0, precision=2)))
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

    move to colorutil?

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
    def __init__(self, color, alpha=None, space=None):
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

        self.color01 = color01

        self.space = space

    def __nice__(self):
        colorpart = ', '.join(['{:.2f}'.format(c) for c in self.color01])
        return self.space + ': ' + colorpart

    def _forimage(self, image, space='rgb'):
        """
        Experimental function.

        Create a numeric color tuple that agrees with the format of the input
        image (i.e. float or int, with 3 or 4 channels).

        Args:
            image (ndarray): image to return color for
            space (str, default=rgb): colorspace of the input image.

        Example:
            >>> img_f3 = np.zeros([8, 8, 3], dtype=np.float32)
            >>> img_u3 = np.zeros([8, 8, 3], dtype=np.uint8)
            >>> img_f4 = np.zeros([8, 8, 4], dtype=np.float32)
            >>> img_u4 = np.zeros([8, 8, 4], dtype=np.uint8)
            >>> Color('red')._forimage(img_f3)
            (1.0, 0.0, 0.0)
            >>> Color('red')._forimage(img_f4)
            (1.0, 0.0, 0.0, 1.0)
            >>> Color('red')._forimage(img_u3)
            (255, 0, 0)
            >>> Color('red')._forimage(img_u4)
            (255, 0, 0, 255)
            >>> Color('red', alpha=0.5)._forimage(img_f4)
            (1.0, 0.0, 0.0, 0.5)
            >>> Color('red', alpha=0.5)._forimage(img_u4)
            (255, 0, 0, 127)
        """
        if im_core.num_channels(image) == 4:
            if not space.endswith('a'):
                space = space + 'a'
        if image.dtype.kind == 'f':
            color = self.as01(space)
        else:
            color = self.as255(space)
        return color

    def ashex(self, space=None):
        c255 = self.as255(space)
        return '#' + ''.join(['{:02x}'.format(c) for c in c255])

    def as255(self, space=None):
        # TODO: be more efficient about not changing to 01 space
        color = tuple(int(c * 255) for c in self.as01(space))
        return color

    def as01(self, space=None):
        """
        self = mplutil.Color('red')
        mplutil.Color('green').as01('rgba')

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
        mplutil.Color._string_to_01('green')
        mplutil.Color._string_to_01('red')
        """
        if color == 'random':
            import random
            color = random.choice(Color.named_colors())

        if color in BASE_COLORS:
            color01 = BASE_COLORS[color]
        elif color in CSS4_COLORS:
            color_hex = CSS4_COLORS[color]
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
            >>> import kwplot
            >>> kwplot.autompl()
            >>> canvas = kwplot.make_legend_img(color_lut)
            >>> kwplot.imshow(canvas)
        """
        names = sorted(list(BASE_COLORS.keys()) + list(CSS4_COLORS.keys()))
        return names

    @classmethod
    def distinct(Color, num, existing=None, space='rgb', legacy='auto',
                 exclude_black=True, exclude_white=True):
        """
        Make multiple distinct colors

        References:
            https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

        Example:
            >>> # xdoctest: +REQUIRES(module:matplotlib)
            >>> from kwimage.im_color import *  # NOQA
            >>> from kwimage.im_color import _draw_color_swatch
            >>> import kwimage
            >>> colors1 = kwimage.Color.distinct(10, legacy=False)
            >>> swatch1 = _draw_color_swatch(colors1, cellshape=9)
            >>> colors2 = kwimage.Color.distinct(10, existing=colors1)
            >>> swatch2 = _draw_color_swatch(colors1 + colors2, cellshape=9)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(swatch1, pnum=(1, 2, 1), fnum=1)
            >>> kwplot.imshow(swatch2, pnum=(1, 2, 2), fnum=1)

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

        if 0:
            import kwimage
            from distinctipy import distinctipy
            existing_colors = kwimage.Color.distinct(5)
            distinctipy.color_swatch(existing_colors)
            # distinctipy.get_colors(10)
            new_colors = distinctipy.get_colors(10, existing_colors)
            distinctipy.color_swatch(existing_colors + new_colors)

    @classmethod
    def random(Color, pool='named'):
        return Color('random')

    def distance(self, other, space='lab'):
        """
        Distance between self an another color

        Example:
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
            print(ub.repr2(ub.sorted_vals(grays), nl=-1))

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


def _draw_color_swatch(colors, cellshape=9):
    """
    Draw colors in a grid
    """
    import kwimage
    import math
    if not ub.iterable(cellshape):
        cellshape = [cellshape, cellshape]
    cell_h = cellshape[0]
    cell_w = cellshape[1]
    cells = []
    for color in colors:
        color_arr = np.array(kwimage.Color(color).as01()).astype(np.float32)
        cell_pixel = color_arr[None, None]
        cell = np.tile(cell_pixel, (cell_h, cell_w, 1))
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


BASE_COLORS = {
    'b': (0, 0, 1),
    'g': (0, 0.5, 0),
    'r': (1, 0, 0),
    'c': (0, 0.75, 0.75),
    'm': (0.75, 0, 0.75),
    'y': (0.75, 0.75, 0),
    'k': (0, 0, 0),
    'w': (1, 1, 1)}


# These colors are from Tableau
TABLEAU_COLORS = (
    ('blue', '#1f77b4'),
    ('orange', '#ff7f0e'),
    ('green', '#2ca02c'),
    ('red', '#d62728'),
    ('purple', '#9467bd'),
    ('brown', '#8c564b'),
    ('pink', '#e377c2'),
    ('gray', '#7f7f7f'),
    ('olive', '#bcbd22'),
    ('cyan', '#17becf'),
)

# Normalize name to "tab:<name>" to avoid name collisions.
TABLEAU_COLORS = OrderedDict(
    ('tab:' + name, value) for name, value in TABLEAU_COLORS)

# This mapping of color names -> hex values is taken from
# a survey run by Randall Munroe see:
# http://blog.xkcd.com/2010/05/03/color-survey-results/
# for more details.  The results are hosted at
# https://xkcd.com/color/rgb.txt
#
# License: http://creativecommons.org/publicdomain/zero/1.0/
XKCD_COLORS = {
    'cloudy blue': '#acc2d9',
    'dark pastel green': '#56ae57',
    'dust': '#b2996e',
    'electric lime': '#a8ff04',
    'fresh green': '#69d84f',
    'light eggplant': '#894585',
    'nasty green': '#70b23f',
    'really light blue': '#d4ffff',
    'tea': '#65ab7c',
    'warm purple': '#952e8f',
    'yellowish tan': '#fcfc81',
    'cement': '#a5a391',
    'dark grass green': '#388004',
    'dusty teal': '#4c9085',
    'grey teal': '#5e9b8a',
    'macaroni and cheese': '#efb435',
    'pinkish tan': '#d99b82',
    'spruce': '#0a5f38',
    'strong blue': '#0c06f7',
    'toxic green': '#61de2a',
    'windows blue': '#3778bf',
    'blue blue': '#2242c7',
    'blue with a hint of purple': '#533cc6',
    'booger': '#9bb53c',
    'bright sea green': '#05ffa6',
    'dark green blue': '#1f6357',
    'deep turquoise': '#017374',
    'green teal': '#0cb577',
    'strong pink': '#ff0789',
    'bland': '#afa88b',
    'deep aqua': '#08787f',
    'lavender pink': '#dd85d7',
    'light moss green': '#a6c875',
    'light seafoam green': '#a7ffb5',
    'olive yellow': '#c2b709',
    'pig pink': '#e78ea5',
    'deep lilac': '#966ebd',
    'desert': '#ccad60',
    'dusty lavender': '#ac86a8',
    'purpley grey': '#947e94',
    'purply': '#983fb2',
    'candy pink': '#ff63e9',
    'light pastel green': '#b2fba5',
    'boring green': '#63b365',
    'kiwi green': '#8ee53f',
    'light grey green': '#b7e1a1',
    'orange pink': '#ff6f52',
    'tea green': '#bdf8a3',
    'very light brown': '#d3b683',
    'egg shell': '#fffcc4',
    'eggplant purple': '#430541',
    'powder pink': '#ffb2d0',
    'reddish grey': '#997570',
    'baby shit brown': '#ad900d',
    'liliac': '#c48efd',
    'stormy blue': '#507b9c',
    'ugly brown': '#7d7103',
    'custard': '#fffd78',
    'darkish pink': '#da467d',
    'deep brown': '#410200',
    'greenish beige': '#c9d179',
    'manilla': '#fffa86',
    'off blue': '#5684ae',
    'battleship grey': '#6b7c85',
    'browny green': '#6f6c0a',
    'bruise': '#7e4071',
    'kelley green': '#009337',
    'sickly yellow': '#d0e429',
    'sunny yellow': '#fff917',
    'azul': '#1d5dec',
    'darkgreen': '#054907',
    'green/yellow': '#b5ce08',
    'lichen': '#8fb67b',
    'light light green': '#c8ffb0',
    'pale gold': '#fdde6c',
    'sun yellow': '#ffdf22',
    'tan green': '#a9be70',
    'burple': '#6832e3',
    'butterscotch': '#fdb147',
    'toupe': '#c7ac7d',
    'dark cream': '#fff39a',
    'indian red': '#850e04',
    'light lavendar': '#efc0fe',
    'poison green': '#40fd14',
    'baby puke green': '#b6c406',
    'bright yellow green': '#9dff00',
    'charcoal grey': '#3c4142',
    'squash': '#f2ab15',
    'cinnamon': '#ac4f06',
    'light pea green': '#c4fe82',
    'radioactive green': '#2cfa1f',
    'raw sienna': '#9a6200',
    'baby purple': '#ca9bf7',
    'cocoa': '#875f42',
    'light royal blue': '#3a2efe',
    'orangeish': '#fd8d49',
    'rust brown': '#8b3103',
    'sand brown': '#cba560',
    'swamp': '#698339',
    'tealish green': '#0cdc73',
    'burnt siena': '#b75203',
    'camo': '#7f8f4e',
    'dusk blue': '#26538d',
    'fern': '#63a950',
    'old rose': '#c87f89',
    'pale light green': '#b1fc99',
    'peachy pink': '#ff9a8a',
    'rosy pink': '#f6688e',
    'light bluish green': '#76fda8',
    'light bright green': '#53fe5c',
    'light neon green': '#4efd54',
    'light seafoam': '#a0febf',
    'tiffany blue': '#7bf2da',
    'washed out green': '#bcf5a6',
    'browny orange': '#ca6b02',
    'nice blue': '#107ab0',
    'sapphire': '#2138ab',
    'greyish teal': '#719f91',
    'orangey yellow': '#fdb915',
    'parchment': '#fefcaf',
    'straw': '#fcf679',
    'very dark brown': '#1d0200',
    'terracota': '#cb6843',
    'ugly blue': '#31668a',
    'clear blue': '#247afd',
    'creme': '#ffffb6',
    'foam green': '#90fda9',
    'grey/green': '#86a17d',
    'light gold': '#fddc5c',
    'seafoam blue': '#78d1b6',
    'topaz': '#13bbaf',
    'violet pink': '#fb5ffc',
    'wintergreen': '#20f986',
    'yellow tan': '#ffe36e',
    'dark fuchsia': '#9d0759',
    'indigo blue': '#3a18b1',
    'light yellowish green': '#c2ff89',
    'pale magenta': '#d767ad',
    'rich purple': '#720058',
    'sunflower yellow': '#ffda03',
    'green/blue': '#01c08d',
    'leather': '#ac7434',
    'racing green': '#014600',
    'vivid purple': '#9900fa',
    'dark royal blue': '#02066f',
    'hazel': '#8e7618',
    'muted pink': '#d1768f',
    'booger green': '#96b403',
    'canary': '#fdff63',
    'cool grey': '#95a3a6',
    'dark taupe': '#7f684e',
    'darkish purple': '#751973',
    'true green': '#089404',
    'coral pink': '#ff6163',
    'dark sage': '#598556',
    'dark slate blue': '#214761',
    'flat blue': '#3c73a8',
    'mushroom': '#ba9e88',
    'rich blue': '#021bf9',
    'dirty purple': '#734a65',
    'greenblue': '#23c48b',
    'icky green': '#8fae22',
    'light khaki': '#e6f2a2',
    'warm blue': '#4b57db',
    'dark hot pink': '#d90166',
    'deep sea blue': '#015482',
    'carmine': '#9d0216',
    'dark yellow green': '#728f02',
    'pale peach': '#ffe5ad',
    'plum purple': '#4e0550',
    'golden rod': '#f9bc08',
    'neon red': '#ff073a',
    'old pink': '#c77986',
    'very pale blue': '#d6fffe',
    'blood orange': '#fe4b03',
    'grapefruit': '#fd5956',
    'sand yellow': '#fce166',
    'clay brown': '#b2713d',
    'dark blue grey': '#1f3b4d',
    'flat green': '#699d4c',
    'light green blue': '#56fca2',
    'warm pink': '#fb5581',
    'dodger blue': '#3e82fc',
    'gross green': '#a0bf16',
    'ice': '#d6fffa',
    'metallic blue': '#4f738e',
    'pale salmon': '#ffb19a',
    'sap green': '#5c8b15',
    'algae': '#54ac68',
    'bluey grey': '#89a0b0',
    'greeny grey': '#7ea07a',
    'highlighter green': '#1bfc06',
    'light light blue': '#cafffb',
    'light mint': '#b6ffbb',
    'raw umber': '#a75e09',
    'vivid blue': '#152eff',
    'deep lavender': '#8d5eb7',
    'dull teal': '#5f9e8f',
    'light greenish blue': '#63f7b4',
    'mud green': '#606602',
    'pinky': '#fc86aa',
    'red wine': '#8c0034',
    'shit green': '#758000',
    'tan brown': '#ab7e4c',
    'darkblue': '#030764',
    'rosa': '#fe86a4',
    'lipstick': '#d5174e',
    'pale mauve': '#fed0fc',
    'claret': '#680018',
    'dandelion': '#fedf08',
    'orangered': '#fe420f',
    'poop green': '#6f7c00',
    'ruby': '#ca0147',
    'dark': '#1b2431',
    'greenish turquoise': '#00fbb0',
    'pastel red': '#db5856',
    'piss yellow': '#ddd618',
    'bright cyan': '#41fdfe',
    'dark coral': '#cf524e',
    'algae green': '#21c36f',
    'darkish red': '#a90308',
    'reddy brown': '#6e1005',
    'blush pink': '#fe828c',
    'camouflage green': '#4b6113',
    'lawn green': '#4da409',
    'putty': '#beae8a',
    'vibrant blue': '#0339f8',
    'dark sand': '#a88f59',
    'purple/blue': '#5d21d0',
    'saffron': '#feb209',
    'twilight': '#4e518b',
    'warm brown': '#964e02',
    'bluegrey': '#85a3b2',
    'bubble gum pink': '#ff69af',
    'duck egg blue': '#c3fbf4',
    'greenish cyan': '#2afeb7',
    'petrol': '#005f6a',
    'royal': '#0c1793',
    'butter': '#ffff81',
    'dusty orange': '#f0833a',
    'off yellow': '#f1f33f',
    'pale olive green': '#b1d27b',
    'orangish': '#fc824a',
    'leaf': '#71aa34',
    'light blue grey': '#b7c9e2',
    'dried blood': '#4b0101',
    'lightish purple': '#a552e6',
    'rusty red': '#af2f0d',
    'lavender blue': '#8b88f8',
    'light grass green': '#9af764',
    'light mint green': '#a6fbb2',
    'sunflower': '#ffc512',
    'velvet': '#750851',
    'brick orange': '#c14a09',
    'lightish red': '#fe2f4a',
    'pure blue': '#0203e2',
    'twilight blue': '#0a437a',
    'violet red': '#a50055',
    'yellowy brown': '#ae8b0c',
    'carnation': '#fd798f',
    'muddy yellow': '#bfac05',
    'dark seafoam green': '#3eaf76',
    'deep rose': '#c74767',
    'dusty red': '#b9484e',
    'grey/blue': '#647d8e',
    'lemon lime': '#bffe28',
    'purple/pink': '#d725de',
    'brown yellow': '#b29705',
    'purple brown': '#673a3f',
    'wisteria': '#a87dc2',
    'banana yellow': '#fafe4b',
    'lipstick red': '#c0022f',
    'water blue': '#0e87cc',
    'brown grey': '#8d8468',
    'vibrant purple': '#ad03de',
    'baby green': '#8cff9e',
    'barf green': '#94ac02',
    'eggshell blue': '#c4fff7',
    'sandy yellow': '#fdee73',
    'cool green': '#33b864',
    'pale': '#fff9d0',
    'blue/grey': '#758da3',
    'hot magenta': '#f504c9',
    'greyblue': '#77a1b5',
    'purpley': '#8756e4',
    'baby shit green': '#889717',
    'brownish pink': '#c27e79',
    'dark aquamarine': '#017371',
    'diarrhea': '#9f8303',
    'light mustard': '#f7d560',
    'pale sky blue': '#bdf6fe',
    'turtle green': '#75b84f',
    'bright olive': '#9cbb04',
    'dark grey blue': '#29465b',
    'greeny brown': '#696006',
    'lemon green': '#adf802',
    'light periwinkle': '#c1c6fc',
    'seaweed green': '#35ad6b',
    'sunshine yellow': '#fffd37',
    'ugly purple': '#a442a0',
    'medium pink': '#f36196',
    'puke brown': '#947706',
    'very light pink': '#fff4f2',
    'viridian': '#1e9167',
    'bile': '#b5c306',
    'faded yellow': '#feff7f',
    'very pale green': '#cffdbc',
    'vibrant green': '#0add08',
    'bright lime': '#87fd05',
    'spearmint': '#1ef876',
    'light aquamarine': '#7bfdc7',
    'light sage': '#bcecac',
    'yellowgreen': '#bbf90f',
    'baby poo': '#ab9004',
    'dark seafoam': '#1fb57a',
    'deep teal': '#00555a',
    'heather': '#a484ac',
    'rust orange': '#c45508',
    'dirty blue': '#3f829d',
    'fern green': '#548d44',
    'bright lilac': '#c95efb',
    'weird green': '#3ae57f',
    'peacock blue': '#016795',
    'avocado green': '#87a922',
    'faded orange': '#f0944d',
    'grape purple': '#5d1451',
    'hot green': '#25ff29',
    'lime yellow': '#d0fe1d',
    'mango': '#ffa62b',
    'shamrock': '#01b44c',
    'bubblegum': '#ff6cb5',
    'purplish brown': '#6b4247',
    'vomit yellow': '#c7c10c',
    'pale cyan': '#b7fffa',
    'key lime': '#aeff6e',
    'tomato red': '#ec2d01',
    'lightgreen': '#76ff7b',
    'merlot': '#730039',
    'night blue': '#040348',
    'purpleish pink': '#df4ec8',
    'apple': '#6ecb3c',
    'baby poop green': '#8f9805',
    'green apple': '#5edc1f',
    'heliotrope': '#d94ff5',
    'yellow/green': '#c8fd3d',
    'almost black': '#070d0d',
    'cool blue': '#4984b8',
    'leafy green': '#51b73b',
    'mustard brown': '#ac7e04',
    'dusk': '#4e5481',
    'dull brown': '#876e4b',
    'frog green': '#58bc08',
    'vivid green': '#2fef10',
    'bright light green': '#2dfe54',
    'fluro green': '#0aff02',
    'kiwi': '#9cef43',
    'seaweed': '#18d17b',
    'navy green': '#35530a',
    'ultramarine blue': '#1805db',
    'iris': '#6258c4',
    'pastel orange': '#ff964f',
    'yellowish orange': '#ffab0f',
    'perrywinkle': '#8f8ce7',
    'tealish': '#24bca8',
    'dark plum': '#3f012c',
    'pear': '#cbf85f',
    'pinkish orange': '#ff724c',
    'midnight purple': '#280137',
    'light urple': '#b36ff6',
    'dark mint': '#48c072',
    'greenish tan': '#bccb7a',
    'light burgundy': '#a8415b',
    'turquoise blue': '#06b1c4',
    'ugly pink': '#cd7584',
    'sandy': '#f1da7a',
    'electric pink': '#ff0490',
    'muted purple': '#805b87',
    'mid green': '#50a747',
    'greyish': '#a8a495',
    'neon yellow': '#cfff04',
    'banana': '#ffff7e',
    'carnation pink': '#ff7fa7',
    'tomato': '#ef4026',
    'sea': '#3c9992',
    'muddy brown': '#886806',
    'turquoise green': '#04f489',
    'buff': '#fef69e',
    'fawn': '#cfaf7b',
    'muted blue': '#3b719f',
    'pale rose': '#fdc1c5',
    'dark mint green': '#20c073',
    'amethyst': '#9b5fc0',
    'blue/green': '#0f9b8e',
    'chestnut': '#742802',
    'sick green': '#9db92c',
    'pea': '#a4bf20',
    'rusty orange': '#cd5909',
    'stone': '#ada587',
    'rose red': '#be013c',
    'pale aqua': '#b8ffeb',
    'deep orange': '#dc4d01',
    'earth': '#a2653e',
    'mossy green': '#638b27',
    'grassy green': '#419c03',
    'pale lime green': '#b1ff65',
    'light grey blue': '#9dbcd4',
    'pale grey': '#fdfdfe',
    'asparagus': '#77ab56',
    'blueberry': '#464196',
    'purple red': '#990147',
    'pale lime': '#befd73',
    'greenish teal': '#32bf84',
    'caramel': '#af6f09',
    'deep magenta': '#a0025c',
    'light peach': '#ffd8b1',
    'milk chocolate': '#7f4e1e',
    'ocher': '#bf9b0c',
    'off green': '#6ba353',
    'purply pink': '#f075e6',
    'lightblue': '#7bc8f6',
    'dusky blue': '#475f94',
    'golden': '#f5bf03',
    'light beige': '#fffeb6',
    'butter yellow': '#fffd74',
    'dusky purple': '#895b7b',
    'french blue': '#436bad',
    'ugly yellow': '#d0c101',
    'greeny yellow': '#c6f808',
    'orangish red': '#f43605',
    'shamrock green': '#02c14d',
    'orangish brown': '#b25f03',
    'tree green': '#2a7e19',
    'deep violet': '#490648',
    'gunmetal': '#536267',
    'blue/purple': '#5a06ef',
    'cherry': '#cf0234',
    'sandy brown': '#c4a661',
    'warm grey': '#978a84',
    'dark indigo': '#1f0954',
    'midnight': '#03012d',
    'bluey green': '#2bb179',
    'grey pink': '#c3909b',
    'soft purple': '#a66fb5',
    'blood': '#770001',
    'brown red': '#922b05',
    'medium grey': '#7d7f7c',
    'berry': '#990f4b',
    'poo': '#8f7303',
    'purpley pink': '#c83cb9',
    'light salmon': '#fea993',
    'snot': '#acbb0d',
    'easter purple': '#c071fe',
    'light yellow green': '#ccfd7f',
    'dark navy blue': '#00022e',
    'drab': '#828344',
    'light rose': '#ffc5cb',
    'rouge': '#ab1239',
    'purplish red': '#b0054b',
    'slime green': '#99cc04',
    'baby poop': '#937c00',
    'irish green': '#019529',
    'pink/purple': '#ef1de7',
    'dark navy': '#000435',
    'greeny blue': '#42b395',
    'light plum': '#9d5783',
    'pinkish grey': '#c8aca9',
    'dirty orange': '#c87606',
    'rust red': '#aa2704',
    'pale lilac': '#e4cbff',
    'orangey red': '#fa4224',
    'primary blue': '#0804f9',
    'kermit green': '#5cb200',
    'brownish purple': '#76424e',
    'murky green': '#6c7a0e',
    'wheat': '#fbdd7e',
    'very dark purple': '#2a0134',
    'bottle green': '#044a05',
    'watermelon': '#fd4659',
    'deep sky blue': '#0d75f8',
    'fire engine red': '#fe0002',
    'yellow ochre': '#cb9d06',
    'pumpkin orange': '#fb7d07',
    'pale olive': '#b9cc81',
    'light lilac': '#edc8ff',
    'lightish green': '#61e160',
    'carolina blue': '#8ab8fe',
    'mulberry': '#920a4e',
    'shocking pink': '#fe02a2',
    'auburn': '#9a3001',
    'bright lime green': '#65fe08',
    'celadon': '#befdb7',
    'pinkish brown': '#b17261',
    'poo brown': '#885f01',
    'bright sky blue': '#02ccfe',
    'celery': '#c1fd95',
    'dirt brown': '#836539',
    'strawberry': '#fb2943',
    'dark lime': '#84b701',
    'copper': '#b66325',
    'medium brown': '#7f5112',
    'muted green': '#5fa052',
    "robin's egg": '#6dedfd',
    'bright aqua': '#0bf9ea',
    'bright lavender': '#c760ff',
    'ivory': '#ffffcb',
    'very light purple': '#f6cefc',
    'light navy': '#155084',
    'pink red': '#f5054f',
    'olive brown': '#645403',
    'poop brown': '#7a5901',
    'mustard green': '#a8b504',
    'ocean green': '#3d9973',
    'very dark blue': '#000133',
    'dusty green': '#76a973',
    'light navy blue': '#2e5a88',
    'minty green': '#0bf77d',
    'adobe': '#bd6c48',
    'barney': '#ac1db8',
    'jade green': '#2baf6a',
    'bright light blue': '#26f7fd',
    'light lime': '#aefd6c',
    'dark khaki': '#9b8f55',
    'orange yellow': '#ffad01',
    'ocre': '#c69c04',
    'maize': '#f4d054',
    'faded pink': '#de9dac',
    'british racing green': '#05480d',
    'sandstone': '#c9ae74',
    'mud brown': '#60460f',
    'light sea green': '#98f6b0',
    'robin egg blue': '#8af1fe',
    'aqua marine': '#2ee8bb',
    'dark sea green': '#11875d',
    'soft pink': '#fdb0c0',
    'orangey brown': '#b16002',
    'cherry red': '#f7022a',
    'burnt yellow': '#d5ab09',
    'brownish grey': '#86775f',
    'camel': '#c69f59',
    'purplish grey': '#7a687f',
    'marine': '#042e60',
    'greyish pink': '#c88d94',
    'pale turquoise': '#a5fbd5',
    'pastel yellow': '#fffe71',
    'bluey purple': '#6241c7',
    'canary yellow': '#fffe40',
    'faded red': '#d3494e',
    'sepia': '#985e2b',
    'coffee': '#a6814c',
    'bright magenta': '#ff08e8',
    'mocha': '#9d7651',
    'ecru': '#feffca',
    'purpleish': '#98568d',
    'cranberry': '#9e003a',
    'darkish green': '#287c37',
    'brown orange': '#b96902',
    'dusky rose': '#ba6873',
    'melon': '#ff7855',
    'sickly green': '#94b21c',
    'silver': '#c5c9c7',
    'purply blue': '#661aee',
    'purpleish blue': '#6140ef',
    'hospital green': '#9be5aa',
    'shit brown': '#7b5804',
    'mid blue': '#276ab3',
    'amber': '#feb308',
    'easter green': '#8cfd7e',
    'soft blue': '#6488ea',
    'cerulean blue': '#056eee',
    'golden brown': '#b27a01',
    'bright turquoise': '#0ffef9',
    'red pink': '#fa2a55',
    'red purple': '#820747',
    'greyish brown': '#7a6a4f',
    'vermillion': '#f4320c',
    'russet': '#a13905',
    'steel grey': '#6f828a',
    'lighter purple': '#a55af4',
    'bright violet': '#ad0afd',
    'prussian blue': '#004577',
    'slate green': '#658d6d',
    'dirty pink': '#ca7b80',
    'dark blue green': '#005249',
    'pine': '#2b5d34',
    'yellowy green': '#bff128',
    'dark gold': '#b59410',
    'bluish': '#2976bb',
    'darkish blue': '#014182',
    'dull red': '#bb3f3f',
    'pinky red': '#fc2647',
    'bronze': '#a87900',
    'pale teal': '#82cbb2',
    'military green': '#667c3e',
    'barbie pink': '#fe46a5',
    'bubblegum pink': '#fe83cc',
    'pea soup green': '#94a617',
    'dark mustard': '#a88905',
    'shit': '#7f5f00',
    'medium purple': '#9e43a2',
    'very dark green': '#062e03',
    'dirt': '#8a6e45',
    'dusky pink': '#cc7a8b',
    'red violet': '#9e0168',
    'lemon yellow': '#fdff38',
    'pistachio': '#c0fa8b',
    'dull yellow': '#eedc5b',
    'dark lime green': '#7ebd01',
    'denim blue': '#3b5b92',
    'teal blue': '#01889f',
    'lightish blue': '#3d7afd',
    'purpley blue': '#5f34e7',
    'light indigo': '#6d5acf',
    'swamp green': '#748500',
    'brown green': '#706c11',
    'dark maroon': '#3c0008',
    'hot purple': '#cb00f5',
    'dark forest green': '#002d04',
    'faded blue': '#658cbb',
    'drab green': '#749551',
    'light lime green': '#b9ff66',
    'snot green': '#9dc100',
    'yellowish': '#faee66',
    'light blue green': '#7efbb3',
    'bordeaux': '#7b002c',
    'light mauve': '#c292a1',
    'ocean': '#017b92',
    'marigold': '#fcc006',
    'muddy green': '#657432',
    'dull orange': '#d8863b',
    'steel': '#738595',
    'electric purple': '#aa23ff',
    'fluorescent green': '#08ff08',
    'yellowish brown': '#9b7a01',
    'blush': '#f29e8e',
    'soft green': '#6fc276',
    'bright orange': '#ff5b00',
    'lemon': '#fdff52',
    'purple grey': '#866f85',
    'acid green': '#8ffe09',
    'pale lavender': '#eecffe',
    'violet blue': '#510ac9',
    'light forest green': '#4f9153',
    'burnt red': '#9f2305',
    'khaki green': '#728639',
    'cerise': '#de0c62',
    'faded purple': '#916e99',
    'apricot': '#ffb16d',
    'dark olive green': '#3c4d03',
    'grey brown': '#7f7053',
    'green grey': '#77926f',
    'true blue': '#010fcc',
    'pale violet': '#ceaefa',
    'periwinkle blue': '#8f99fb',
    'light sky blue': '#c6fcff',
    'blurple': '#5539cc',
    'green brown': '#544e03',
    'bluegreen': '#017a79',
    'bright teal': '#01f9c6',
    'brownish yellow': '#c9b003',
    'pea soup': '#929901',
    'forest': '#0b5509',
    'barney purple': '#a00498',
    'ultramarine': '#2000b1',
    'purplish': '#94568c',
    'puke yellow': '#c2be0e',
    'bluish grey': '#748b97',
    'dark periwinkle': '#665fd1',
    'dark lilac': '#9c6da5',
    'reddish': '#c44240',
    'light maroon': '#a24857',
    'dusty purple': '#825f87',
    'terra cotta': '#c9643b',
    'avocado': '#90b134',
    'marine blue': '#01386a',
    'teal green': '#25a36f',
    'slate grey': '#59656d',
    'lighter green': '#75fd63',
    'electric green': '#21fc0d',
    'dusty blue': '#5a86ad',
    'golden yellow': '#fec615',
    'bright yellow': '#fffd01',
    'light lavender': '#dfc5fe',
    'umber': '#b26400',
    'poop': '#7f5e00',
    'dark peach': '#de7e5d',
    'jungle green': '#048243',
    'eggshell': '#ffffd4',
    'denim': '#3b638c',
    'yellow brown': '#b79400',
    'dull purple': '#84597e',
    'chocolate brown': '#411900',
    'wine red': '#7b0323',
    'neon blue': '#04d9ff',
    'dirty green': '#667e2c',
    'light tan': '#fbeeac',
    'ice blue': '#d7fffe',
    'cadet blue': '#4e7496',
    'dark mauve': '#874c62',
    'very light blue': '#d5ffff',
    'grey purple': '#826d8c',
    'pastel pink': '#ffbacd',
    'very light green': '#d1ffbd',
    'dark sky blue': '#448ee4',
    'evergreen': '#05472a',
    'dull pink': '#d5869d',
    'aubergine': '#3d0734',
    'mahogany': '#4a0100',
    'reddish orange': '#f8481c',
    'deep green': '#02590f',
    'vomit green': '#89a203',
    'purple pink': '#e03fd8',
    'dusty pink': '#d58a94',
    'faded green': '#7bb274',
    'camo green': '#526525',
    'pinky purple': '#c94cbe',
    'pink purple': '#db4bda',
    'brownish red': '#9e3623',
    'dark rose': '#b5485d',
    'mud': '#735c12',
    'brownish': '#9c6d57',
    'emerald green': '#028f1e',
    'pale brown': '#b1916e',
    'dull blue': '#49759c',
    'burnt umber': '#a0450e',
    'medium green': '#39ad48',
    'clay': '#b66a50',
    'light aqua': '#8cffdb',
    'light olive green': '#a4be5c',
    'brownish orange': '#cb7723',
    'dark aqua': '#05696b',
    'purplish pink': '#ce5dae',
    'dark salmon': '#c85a53',
    'greenish grey': '#96ae8d',
    'jade': '#1fa774',
    'ugly green': '#7a9703',
    'dark beige': '#ac9362',
    'emerald': '#01a049',
    'pale red': '#d9544d',
    'light magenta': '#fa5ff7',
    'sky': '#82cafc',
    'light cyan': '#acfffc',
    'yellow orange': '#fcb001',
    'reddish purple': '#910951',
    'reddish pink': '#fe2c54',
    'orchid': '#c875c4',
    'dirty yellow': '#cdc50a',
    'orange red': '#fd411e',
    'deep red': '#9a0200',
    'orange brown': '#be6400',
    'cobalt blue': '#030aa7',
    'neon pink': '#fe019a',
    'rose pink': '#f7879a',
    'greyish purple': '#887191',
    'raspberry': '#b00149',
    'aqua green': '#12e193',
    'salmon pink': '#fe7b7c',
    'tangerine': '#ff9408',
    'brownish green': '#6a6e09',
    'red brown': '#8b2e16',
    'greenish brown': '#696112',
    'pumpkin': '#e17701',
    'pine green': '#0a481e',
    'charcoal': '#343837',
    'baby pink': '#ffb7ce',
    'cornflower': '#6a79f7',
    'blue violet': '#5d06e9',
    'chocolate': '#3d1c02',
    'greyish green': '#82a67d',
    'scarlet': '#be0119',
    'green yellow': '#c9ff27',
    'dark olive': '#373e02',
    'sienna': '#a9561e',
    'pastel purple': '#caa0ff',
    'terracotta': '#ca6641',
    'aqua blue': '#02d8e9',
    'sage green': '#88b378',
    'blood red': '#980002',
    'deep pink': '#cb0162',
    'grass': '#5cac2d',
    'moss': '#769958',
    'pastel blue': '#a2bffe',
    'bluish green': '#10a674',
    'green blue': '#06b48b',
    'dark tan': '#af884a',
    'greenish blue': '#0b8b87',
    'pale orange': '#ffa756',
    'vomit': '#a2a415',
    'forrest green': '#154406',
    'dark lavender': '#856798',
    'dark violet': '#34013f',
    'purple blue': '#632de9',
    'dark cyan': '#0a888a',
    'olive drab': '#6f7632',
    'pinkish': '#d46a7e',
    'cobalt': '#1e488f',
    'neon purple': '#bc13fe',
    'light turquoise': '#7ef4cc',
    'apple green': '#76cd26',
    'dull green': '#74a662',
    'wine': '#80013f',
    'powder blue': '#b1d1fc',
    'off white': '#ffffe4',
    'electric blue': '#0652ff',
    'dark turquoise': '#045c5a',
    'blue purple': '#5729ce',
    'azure': '#069af3',
    'bright red': '#ff000d',
    'pinkish red': '#f10c45',
    'cornflower blue': '#5170d7',
    'light olive': '#acbf69',
    'grape': '#6c3461',
    'greyish blue': '#5e819d',
    'purplish blue': '#601ef9',
    'yellowish green': '#b0dd16',
    'greenish yellow': '#cdfd02',
    'medium blue': '#2c6fbb',
    'dusty rose': '#c0737a',
    'light violet': '#d6b4fc',
    'midnight blue': '#020035',
    'bluish purple': '#703be7',
    'red orange': '#fd3c06',
    'dark magenta': '#960056',
    'greenish': '#40a368',
    'ocean blue': '#03719c',
    'coral': '#fc5a50',
    'cream': '#ffffc2',
    'reddish brown': '#7f2b0a',
    'burnt sienna': '#b04e0f',
    'brick': '#a03623',
    'sage': '#87ae73',
    'grey green': '#789b73',
    'white': '#ffffff',
    "robin's egg blue": '#98eff9',
    'moss green': '#658b38',
    'steel blue': '#5a7d9a',
    'eggplant': '#380835',
    'light yellow': '#fffe7a',
    'leaf green': '#5ca904',
    'light grey': '#d8dcd6',
    'puke': '#a5a502',
    'pinkish purple': '#d648d7',
    'sea blue': '#047495',
    'pale purple': '#b790d4',
    'slate blue': '#5b7c99',
    'blue grey': '#607c8e',
    'hunter green': '#0b4008',
    'fuchsia': '#ed0dd9',
    'crimson': '#8c000f',
    'pale yellow': '#ffff84',
    'ochre': '#bf9005',
    'mustard yellow': '#d2bd0a',
    'light red': '#ff474c',
    'cerulean': '#0485d1',
    'pale pink': '#ffcfdc',
    'deep blue': '#040273',
    'rust': '#a83c09',
    'light teal': '#90e4c1',
    'slate': '#516572',
    'goldenrod': '#fac205',
    'dark yellow': '#d5b60a',
    'dark grey': '#363737',
    'army green': '#4b5d16',
    'grey blue': '#6b8ba4',
    'seafoam': '#80f9ad',
    'puce': '#a57e52',
    'spring green': '#a9f971',
    'dark orange': '#c65102',
    'sand': '#e2ca76',
    'pastel green': '#b0ff9d',
    'mint': '#9ffeb0',
    'light orange': '#fdaa48',
    'bright pink': '#fe01b1',
    'chartreuse': '#c1f80a',
    'deep purple': '#36013f',
    'dark brown': '#341c02',
    'taupe': '#b9a281',
    'pea green': '#8eab12',
    'puke green': '#9aae07',
    'kelly green': '#02ab2e',
    'seafoam green': '#7af9ab',
    'blue green': '#137e6d',
    'khaki': '#aaa662',
    'burgundy': '#610023',
    'dark teal': '#014d4e',
    'brick red': '#8f1402',
    'royal purple': '#4b006e',
    'plum': '#580f41',
    'mint green': '#8fff9f',
    'gold': '#dbb40c',
    'baby blue': '#a2cffe',
    'yellow green': '#c0fb2d',
    'bright purple': '#be03fd',
    'dark red': '#840000',
    'pale blue': '#d0fefe',
    'grass green': '#3f9b0b',
    'navy': '#01153e',
    'aquamarine': '#04d8b2',
    'burnt orange': '#c04e01',
    'neon green': '#0cff0c',
    'bright blue': '#0165fc',
    'rose': '#cf6275',
    'light pink': '#ffd1df',
    'mustard': '#ceb301',
    'indigo': '#380282',
    'lime': '#aaff32',
    'sea green': '#53fca1',
    'periwinkle': '#8e82fe',
    'dark pink': '#cb416b',
    'olive green': '#677a04',
    'peach': '#ffb07c',
    'pale green': '#c7fdb5',
    'light brown': '#ad8150',
    'hot pink': '#ff028d',
    'black': '#000000',
    'lilac': '#cea2fd',
    'navy blue': '#001146',
    'royal blue': '#0504aa',
    'beige': '#e6daa6',
    'salmon': '#ff796c',
    'olive': '#6e750e',
    'maroon': '#650021',
    'bright green': '#01ff07',
    'dark purple': '#35063e',
    'mauve': '#ae7181',
    'forest green': '#06470c',
    'aqua': '#13eac9',
    'cyan': '#00ffff',
    'tan': '#d1b26f',
    'dark blue': '#00035b',
    'lavender': '#c79fef',
    'turquoise': '#06c2ac',
    'dark green': '#033500',
    'violet': '#9a0eea',
    'light purple': '#bf77f6',
    'lime green': '#89fe05',
    'grey': '#929591',
    'sky blue': '#75bbfd',
    'yellow': '#ffff14',
    'magenta': '#c20078',
    'light green': '#96f97b',
    'orange': '#f97306',
    'teal': '#029386',
    'light blue': '#95d0fc',
    'red': '#e50000',
    'brown': '#653700',
    'pink': '#ff81c0',
    'blue': '#0343df',
    'green': '#15b01a',
    'purple': '#7e1e9c'}

# Normalize name to "xkcd:<name>" to avoid name collisions.
XKCD_COLORS = {'xkcd:' + name: value for name, value in XKCD_COLORS.items()}


# https://drafts.csswg.org/css-color-4/#named-colors
CSS4_COLORS = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkgrey':             '#A9A9A9',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkslategrey':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dimgrey':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'grey':                 '#808080',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgray':            '#D3D3D3',
    'lightgreen':           '#90EE90',
    'lightgrey':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightslategrey':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'rebeccapurple':        '#663399',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#F4A460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'slategrey':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}
