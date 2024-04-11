class Base1(object):
    def __init__(self):
        self.attr1 = True


class Base2(object):
    class_attr1 = True
    def __init__(self):
        self.attr1 = True


class Derived2(Base2):
    class_attr1 = True


def bench_isinstance_vs_attr():
    instances = {
        'base1': Base1(),
        'base2': Base2(),
        'derived2': Derived2(),
    }

    import ubelt as ub
    ti = ub.Timerit(100000, bestof=500, verbose=1, unit='us')

    # Do this twice, but keep the second measure
    data = ub.AutoDict()

    for selfname, self in instances.items():

        print(ub.color_text('--- SELF = {} ---'.format(selfname), 'blue'))

        subdata = data[selfname] = {}

        for timer in ti.reset('isinstance(self, Base1)'):
            with timer:
                isinstance(self, Base1)
        subdata[ti.label] = ti.min()

        for timer in ti.reset('isinstance(self, Base2)'):
            with timer:
                isinstance(self, Base2)
        subdata[ti.label] = ti.min()

        for timer in ti.reset('isinstance(self, Derived2)'):
            with timer:
                isinstance(self, Derived2)
        subdata[ti.label] = ti.min()

        for timer in ti.reset('getattr(self, "class_attr1", False)'):
            with timer:
                getattr(self, 'class_attr1', False)
        subdata[ti.label] = ti.min()

        for timer in ti.reset('getattr(self, "attr1", False)'):
            with timer:
                getattr(self, 'attr1', False)
        subdata[ti.label] = ti.min()

    try:
        import pandas as pd
        df = pd.DataFrame(data) * 1e9
        try:
            from kwil.util.util_pandas import _to_string_monkey
            print(_to_string_monkey(df, key='minima'))
        except Exception:
            print(df)
    except ImportError:
        print('no pandas')
        print(ub.urepr(data, nl=2, precision=4))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwimage/dev/bench_isinstance.py
    """
    bench_isinstance_vs_attr()
