"""
Test NITF Images:
    https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html

"""
from os.path import basename
from os.path import splitext
import ssl
import ubelt as ub
import urllib


def have_gov_certs():
    try:
        # Test to see if we have certs
        ub.download('https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html')
        return True
    except urllib.request.URLError:
        return False


def description():
    import bs4
    import requests
    resp = requests.get('https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/scen_2_1.html', verify=False)
    soup = bs4.BeautifulSoup(resp.text, 'html.parser')
    tables = soup.findAll('table')

    names_noext = [n.split('.')[0] for n in NITF_TEST_NAMES]

    name = None
    name_to_desc = {}

    for tab in tables:
        for td in tab.findAll('td'):
            if name is not None:
                desc = td.text.strip()
                name_to_desc[name] = desc.replace('\r', '').replace('\n', '').replace('\t', '').replace('\xa0', '')
                name = None
            elif td.text.strip() in names_noext:
                name = td.text.strip()
    print(ub.urepr(name_to_desc, nl=1))


NITF_TEST_NAMES = [
    'i_3001a.ntf', 'ns3004f.nsf', 'i_3004g.ntf', 'ns3005b.nsf',
    'i_3008a.ntf', 'ns3010a.nsf', 'i_3015a.ntf', 'ns3017a.nsf',
    'i_3018a.ntf', 'ns3022b.nsf', 'i_3025b.ntf', 'ns3033b.nsf',
    'i_3034c.ntf', 'ns3034d.nsf', 'i_3034f.ntf', 'ns3038a.nsf',
    'i_3041a.ntf', 'ns3050a.nsf', 'i_3051e.ntf', 'ns3051v.nsf',
    'i_3052a.ntf', 'ns3059a.nsf', 'i_3060a.ntf', 'ns3061a.nsf',
    'i_3063f.ntf', 'ns3063h.nsf', 'i_3068a.ntf', 'ns3073a.nsf',
    'i_3076a.ntf', 'ns3090i.nsf', 'i_3090m.ntf', 'ns3090q.nsf',
    'i_3090u.ntf', 'ns3101b.nsf', 'i_3113g.ntf', 'ns3114a.nsf',
    'i_3114e.ntf', 'ns3114i.nsf', 'i_3117ax.ntf', 'ns3118b.nsf',
    'ns3119b.nsf', 'i_3128b.ntf', 'ns3201a.nsf', 'i_3201c.ntf',
    'ns3228b.nsf', 'i_3228c.ntf', 'ns3228d.nsf', 'i_3228e.ntf',
    'ns3229b.nsf', 'i_3301a.ntf', 'ns3301b.nsf', 'i_3301c.ntf',
    'ns3301e.nsf', 'i_3301h.ntf', 'ns3301j.nsf', 'i_3301k.ntf',
    'ns3302a.nsf', 'i_3303a.ntf', 'ns3304a.nsf', 'i_3309a.ntf',
    'ns3310a.nsf', 'i_3311a.ntf', 'ns3321a.nsf', 'ns3361c.nsf',
    'i_3405a.ntf', 'ns3417c.nsf', 'i_3430a.ntf', 'ns3437a.nsf',
    'i_3450c.ntf', 'ns3450e.nsf', 'i_5012c.ntf', 'ns5600a.nsf'
]

NITF_DESC = {
    'i_3001a': 'Can the system handle an uncompressed 1024x1024 8-bit mono image and file contains GEO data? (AIRFIELD)',
    'i_3004g': 'Checks a system to see how it applies GEO data around 00, 180.',
    'i_3008a': 'Checks a JPEG-compressed, 256x256 8-bit mono image, Q4, COMRAT 00.4 with general purpose tables embedded. File also contains image comments. (TANK)',
    'i_3015a': 'Can the system handle a JPEG-compressed 256x256 8-bit mono image with comment in the JPEG stream before frame marker? (TANK)',
    'i_3018a': 'Checks a JPEG-compressed 231x191 8-bit mono image with a corrupted restart marker occurring too early. (BLIMP)',
    'i_3025b': 'Checks to see if a viewer can read a JPEG stream with fill bytes (FF) in the JPEG stream before FFD8. (LINCOLN)',
    'i_3034c': 'Checks a 1-bit RGB/LUT with an arrow, the value of 1 mapped to green and the background value of 0 mapped to red, and no mask table.',
    'i_3034f': 'Checks a 1-bit RGB/LUT (green arrow) with a mask table (pad pixels having value of 0x00) and a transparent pixel value of 1 being mapped to green by the LUT.',
    'i_3041a': 'Checks a bi-level compressed at 2DS 512x512 FAX image. (SHIP)',
    'i_3051e': 'Checks to see if a system can render CGM Text in the proper location.',
    'i_3052a': 'Checks to see if the system renders a basic Circle.',
    'i_3060a': 'Checks for rendering CGM polylines (types 1 through 5.)',
    'i_3063f': 'Checks for rendering CGM polygons with hatch style 5.',
    'i_3068a': 'Checks for rendering CGM rectangles with starting point in Lower Right of rectangle.',
    'i_3076a': 'Checks for rendering various CGM elliptical arc cords.',
    'i_3090m': 'CIRARCC5 checks for proper interpretation of upper left VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'i_3090u': 'CIRARCCD checks for proper interpretation of upper right VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'i_3113g': 'Can system display a Low Bite Rate (LBR) file with an uncompressed image overlay?',
    'i_3114e': 'Checks to see if the system recognizes all UT1 values 0xA0 to 0xFF.',
    'i_3117ax': 'Can the system render an NSIF file having the maximum total bytes in 32 text segments each of 99,998 bytes with an image segment? (Text shows 1 of 32 identical text segments.)',
    'i_3128b': 'This file contains PIAE TREs version 2.0 to include three PEA TREs. If the system supports PIAE TREs, can they find each TRE to include all 3 PEA TREs?',
    'i_3201c': 'Checks a systems ability to handle a single block IMODE R image, 126x126',
    'i_3228c': 'MS IMODE P RGB, multi-blocked image, not all bands displayed.',
    'i_3228e': 'MS IMODE R RGB, multi-blocked image, not all bands displayed.',
    'i_3301a': 'Checks an uncompressed 1024x1024 24-bit multi-blocked (IMode-S) color image. (HELO)',
    'i_3301c': 'Checks an IMODE S image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x00 displaying as transparent, 3x3 blocks.',
    'i_3301h': 'Can the system display a multi block 6x6 IMODE R image and 216x216?',
    'i_3301k': 'Checks an IMODE R image with a data mask subheader, with padded pixels, a pad pixel value of 0x00 displaying as transparent, and 3x3 blocks.',
    'i_3303a': 'Can the system display an uncompressed 2048x2048 8-bit multi-blocked mono image? (CAMELS)',
    'i_3309a': 'Can the system display a JPEG-compressed 2048x2048 8-bit multi-blocked (256x256) mono image w/QFAC=3, RSTI=16, and IMODE=B? (CAMELS)',
    'i_3311a': 'Can the system display a JPEG 2048x2048 24-bit PI block color w/QFAC=3,RSTI=32,IMODE=P, blocked (512x512)? (JET)',
    'i_3405a': 'Can the system handle a multi-blocked 1024x1024 image with 11/16 (ABPP=11, NBPP=16)? (AIRSTRIP)',
    'i_3430a': 'Can the system handle an NSIF file with an uncompressed image with 12-bit back to back data, ABPP = 12, and NBPP = 12?',
    'i_3450c': 'Can the system read a 32-bit real image?',
    'i_5012c': 'Can the system handle an NSIF file with 100 images, 100 symbols and 32 text elements, images 1, 25, 50, 75 and 100 attached to "000", symbol 12 and text 29 attached to image 25, symbol 32 and text 30 attached to image 50, symbol 86 and text 31 attached to image 75, symbol 90 and text 32 attached to image 100, and all other segments attached to image 1?',
    'ns3004f': 'Checks a system to see how it applies GEO data around 00, 000.',
    'ns3005b': 'Checks a JPEG-compressed 1024x1024 8-bit mono image compressed with visible 8-bit tables and COMRAT 01.1. (AIRFIELD)',
    'ns3010a': 'Can the system handle a JPEG-compressed 231x191 8-bit mono image that is non-divide by 8, and file also contains image comments? (BLIMP)',
    'ns3017a': 'Checks a JPEG-compressed 231x191 8-bit mono image with a corrupted restart marker occurring too late. (BLIMP)',
    'ns3022b': 'Checks a JPEG-compressed 181 x 73 8-bit mono image with split Huffman tables 1 DC 1 AC having separate marker for each. (JET)',
    'ns3033b': 'Checks a JPEG-compressed 512x512 8-bit mono image with APP7 marker in JPEG stream. (LENNA)',
    'ns3034d': 'Checks a 1-bit mono with mask table having (0x00) black as transparent with white arrow.',
    'ns3038a': 'Checks all run lengths on a bi-level compressed at 1D and 1024x1024 FAX imagery. (SEMAPHORE)',
    'ns3050a': 'Checks all run lengths on a bi-level compressed at 2DH and 1024x1024 FAX imagery. (SEMAPHORE)',
    'ns3051v': 'Checks to see if the system can render CGM polygon sets properly and two polygons that do not intersect.',
    'ns3059a': 'Checks for rendering CGM ellipses with edge width of 50.',
    'ns3061a': 'Checks an IMODE S image with a data mask subheader, the subheader with padded pixels, having a color value of 0x00, 0x00, 0x00 displaying as transparent, and 3x3 blocks.',
    'ns3063h': 'Checks for rendering CGM polygons with hatch style 1 with auxiliary color.',
    'ns3073a': 'Checks for rendering various CGM circular arcs.',
    'ns3090i': 'CIRARCC1 checks for proper interpretation of lower left VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'ns3090q': 'CIRARCC9 checks for proper interpretation of lower right VDC and drawing of center-closed CGM circular arcs across different quadrants.',
    'ns3101b': 'Checks to see what CGM fonts are supported by the system. The display image is shown with limited font support.',
    'ns3114a': 'Can the render an NSIF file with a single (STA) text segment with only one byte of data?',
    'ns3114i': 'Can the system render a U8S character set (this text segment is in an HTML format)? (To verify data, ensure your web browser is set to properly display Unicode UT8-F.)',
    'ns3118b': 'Can the system render an embedded MTF file is the second text segment. Text shows MTF text segment.Can the system render an embedded MTF file that is the second text segment? (Text shows MTF text segment.)',
    'ns3119b': 'Can the system render the maximum CGM total bytes for a clevel 3 file (total bytes 1,048,576 in 8 CGM segments)?',
    'ns3201a': 'Checks a systems ability to handle an RGB/LUT. (LUT has 128 entries.)',
    'ns3228b': 'MS IMODE S RGB, multi-blocked image, not all bands displayed.',
    'ns3228d': 'MS IMODE B RGB, multi-blocked image, not all bands displayed.',
    'ns3229b': 'Nine band MS image, PVTYPE=SI, ABPP=16 in NBPP=16, IMODE B. Band 1, 2 & 3 have been enhanced for viewing, image is naturally dark.',
    'ns3301b': 'Checks an IMODE B image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x00 displaying as transparent, 3x3 blocks.',
    'ns3301e': 'Checks an IMODE P image with a data mask subheader, the subheader with padded pixels, having a pad pixel value of 0x7F displaying as determined by the ELT, 4x4 blocks.',
    'ns3301j': 'Can the system display a mono JPEG image with mask blocks?',
    'ns3302a': 'Can the system display an uncompressed 256x256 24-bit multi-blocked (IMode-B) image? (TRACKER)',
    'ns3304a': 'Can the system display a JPEG-compressed 2048x2048 8-bit multi-blocked (512x512) mono image w/QFAC=3, RSTI=32, and IMODE=B? (CAMELS)',
    'ns3310a': 'Can the system display an uncompressed, 244x244 24-bit IMODE P multi-blocked (128x128) color image? (BIRDS)',
    'ns3321a': 'Can the system handle an NSIF file containing a streaming file header (in which the image size was unknown at the time of production) and the main header has replacement data?',
    'ns3361c': 'How does the system handle multi-images with GEO data?',
    'ns3417c': 'Can the system handle a 98x208 mono image with custom 12-bit JPEG SAR tables and COMRAT 03.5?',
    'ns3437a': 'Can the system handle a 12-bit JPEG C5 (Lossless) ES implementation multi-blocked 1024x2048 image with APP6 in each displayable block?',
    'ns3450e': 'Can the system read a 64-bit real image?',
    'ns5600a': 'Can the system handle a MS, 31 Band image, 42 by 42 pixels, 32bpp Float, and IREPBANDS all blank?',
}


def grab_nitfs():
    base = 'https://gwg.nga.mil/ntb/baseline/software/testfile/Nitfv2_1/'
    urls = [base + fname for fname in NITF_TEST_NAMES]

    nitf_fpaths = []
    for url in urls:
        fpath = ub.grabdata(url)
        nitf_fpaths.append(fpath)
    return nitf_fpaths


def unsafe_grab_nitfs():
    try:
        print('TRYING TO GRAB DATA SAFELY')
        nitf_fpaths = grab_nitfs()
    except urllib.request.URLError:
        # Very dangerous, probably best to just install the certs, but I'm lazy.
        print('SAFE GRAB FAILED. FALLBACK TO UNSAFE GRAB (IGNORE SSL CERTS)')
        _orig_context = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            nitf_fpaths = grab_nitfs()
        finally:
            # Restore ssl context if we hacked it
            ssl._create_default_https_context = _orig_context
            print('RESTORED SSL CONTEXT')
    else:
        print('GOT DATA SAFELY')
    return nitf_fpaths


def check_nitfs():
    nitfs = unsafe_grab_nitfs()

    import xdev
    import netharn as nh

    total = 0
    for fpath in nitfs:
        nbytes = nh.util.get_file_info(fpath)['filesize']
        print('nbytes = {!r}'.format(xdev.byte_str(nbytes)))
        total += nbytes
    print(xdev.byte_str(total))

    failed_fpaths = []
    passed_fpaths = []

    for fpath in nitfs:
        import kwimage
        try:
            kwimage.imread(fpath)
            passed_fpaths.append(fpath)
        except Exception:
            failed_fpaths.append(fpath)

    print('passed = {}'.format(len(passed_fpaths)))
    print('failed = {}'.format(len(failed_fpaths)))

    print('CANT HANDLE')
    for fpath in failed_fpaths:
        name = splitext(basename(fpath))[0]
        desc = NITF_DESC[name]
        print(desc)

    for fpath in failed_fpaths:
        print('\n-----')
        print('fpath = {!r}'.format(fpath))
        try:
            kwimage.imread(fpath)
        except Exception:
            pass
        print('\n-----')

    from ndsampler.abstract_frames import _cog_cache_write
    for gpath in passed_fpaths:
        cache_gpath = ub.augpath(gpath, ext='.test.api.cog')
        ub.delete(cache_gpath)
        # config = {'hack_use_cli': True}
        config = {'hack_use_cli': False, 'compress': 'LZW'}
        _cog_cache_write(gpath, cache_gpath, config=config)

    from ndsampler.abstract_frames import _cog_cache_write
    for gpath in passed_fpaths:
        cache_gpath = ub.augpath(gpath, ext='.test.cli.cog')
        ub.delete(cache_gpath)
        # config = {'hack_use_cli': True}
        config = {'hack_use_cli': True, 'compress': 'LZW'}
        _cog_cache_write(gpath, cache_gpath, config=config)

    from ndsampler.abstract_frames import _cog_cache_write
    for gpath in passed_fpaths:
        cache_gpath = ub.augpath(gpath, ext='.test.cli.cog')
        ub.delete(cache_gpath)
        # config = {'hack_use_cli': True}
        kwimage.imread(gpath)
        config = {'hack_use_cli': True, 'compress': 'JPEG'}
        _cog_cache_write(gpath, cache_gpath, config=config)
