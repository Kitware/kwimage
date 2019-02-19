from __future__ import absolute_import

cdef extern from "SetCover.h":
    cdef void exec_setcover(unsigned int ** sets,
                   unsigned short ** weights,
                   unsigned int * set_sizes,
                   unsigned int * element_size_lookup,
                   unsigned int set_count,
                   unsigned int uniqu_element_count,
                   unsigned int all_element_count,
                   unsigned int max_weight)


cdef void wrap_exec_set_cover(unsigned int ** sets,
                         unsigned short ** weights,
                         unsigned int * set_sizes,
                         unsigned int * element_size_lookup,
                         unsigned int set_count,
                         unsigned int uniqu_element_count,
                         unsigned int all_element_count,
                         unsigned int max_weight,
                         unsigned int *outvar_keep):
    exec_setcover(sets, weights, set_sizes, element_size, set_count,
        uniqu_element_count all_element_count max_weight, outvar_keep)



def setcover(candidate_sets_dict):
    """
    >>> candidate_sets_dict = {
    >>>     'a': [1, 2, 3, 8, 9, 0],
    >>>     'b': [1, 2, 3, 4, 5],
    >>>     'c': [4, 5, 7],
    >>>     'd': [5, 6, 7],
    >>>     'e': [6, 7, 8, 9, 0],
    >>> }

    """
    import itertools as it
    items = sorted(set(it.chain(*candidate_sets_dict.values())))
    set_keys = sorted(candidate_sets_dict.keys())
    
    item_to_encoded = dict(zip(items, range(len(items))))
    item_to_weight = {item: 1 for item in items}

    items = sorted(candidate_sets_dict.keys())

    sets_ = []
    weights_ = []
    set_sizes_ = []

    set_count = len(item_to_encoded)

    for key in set_keys:
        coverset = candidate_sets_dict[key]
        sets_.append([item_to_encoded[item] for item in coverset])
        weights_.append([item_to_weight[item] for item in coverset])
        set_sizes_.append(len(sets_[-1]))
