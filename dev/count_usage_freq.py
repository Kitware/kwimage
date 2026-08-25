from __future__ import annotations

from collections.abc import Sequence

import kwconf


class UsageConfig(kwconf.Config):
    modname: str = kwconf.Value(required=True)
    print_packages: bool = kwconf.Flag(False)
    remove_zeros: bool = kwconf.Flag(True)
    extra_modnames: list[str] = kwconf.Value(
        default_factory=list,
        parser='csv',
    )


def count_usage(
    cmdline: bool | Sequence[str] | str = True,
    **kw: object,
):
    config = UsageConfig()
    config.load(data=kw, argv=cmdline, strict=True)

    import glob
    from os.path import join

    import ubelt as ub

    names = [
        'netharn',
        'ndsampler',
        'kwimage',
        'kwplot',
        'kwcoco',
    ] + config.extra_modnames

    all_fpaths = []
    for name in names:
        if name:
            repo_fpath = ub.expandpath(join('~/code', name))
            fpaths = glob.glob(join(repo_fpath, '**', '*.py'), recursive=True)
            for fpath in fpaths:
                all_fpaths.append((name, fpath))

    print('names = {}'.format(ub.urepr(names)))

    import re

    module = ub.import_module_from_name(config.modname)

    package_name = module.__name__
    package_allvar = module.__all__

    pat = re.compile(
        r'\b' + package_name + r'\.(?P<attr>[a-zA-Z_][A-Za-z_0-9]*)\b'
    )

    pkg_to_hist = ub.ddict(lambda: ub.ddict(int))
    for name, fpath in ub.ProgIter(all_fpaths):
        text = ub.readfrom(fpath, verbose=0)
        for match in pat.finditer(text):
            attr = match.groupdict()['attr']
            if attr in package_allvar:
                pkg_to_hist[name][attr] += 1

    hist_iter = iter(pkg_to_hist.values())
    usage = next(hist_iter).copy()
    for other in hist_iter:
        for k, v in other.items():
            usage[k] += v
    for attr in package_allvar:
        usage[attr] += 0

    for name in pkg_to_hist.keys():
        pkg_to_hist[name] = ub.odict(
            sorted(pkg_to_hist[name].items(), key=lambda t: t[1])[::-1]
        )

    usage = ub.odict(sorted(usage.items(), key=lambda t: t[1])[::-1])

    if config.print_packages:
        print(ub.urepr(pkg_to_hist, nl=2))

    if config.remove_zeros:
        for k, v in list(usage.items()):
            if v == 0:
                usage.pop(k)

    print(ub.urepr(usage, nl=1))
    return config, usage


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/count_usage_freq.py --help
        python ~/code/kwimage/dev/count_usage_freq.py --modname=kwarray --extra_modnames=bioharn
        python ~/code/kwimage/dev/count_usage_freq.py --modname=kwimage --extra_modnames=bioharn,watch
    """
    count_usage()
