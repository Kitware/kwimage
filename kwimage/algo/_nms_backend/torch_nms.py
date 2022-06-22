import numpy as np
try:
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion


try:
    import torch
except Exception:
    torch = None
    _TORCH_HAS_BOOL_COMP = False
else:
    _TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


def torch_nms(ltrb, scores, classes=None, thresh=.5, bias=0, fast=False):
    """
    Non maximum suppression implemented with pytorch tensors

    CURRENTLY NOT WORKING

    Args:
        ltrb (Tensor): Bounding boxes of one image in the format (ltrb)
        scores (Tensor): Scores of each box
        classes (Tensor | None):
            the classes of each box. If specified nms is applied to each class separately.
        thresh (float): iou threshold

    Returns:
        torch.ByteTensor:
            keep - boolean array indicating which boxes were not pruned.

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> import numpy as np
        >>> ltrb = torch.FloatTensor(np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 130, 130],
        >>>     [100, 100, 130, 130],
        >>>     [100, 100, 130, 130],
        >>> ], dtype=np.float32))
        >>> scores = torch.FloatTensor(np.array([.1, .5, .9, .1, .3, .5, .4]))
        >>> classes = torch.LongTensor(np.array([0, 0, 0, 0, 0, 0, 0]))
        >>> thresh = .5
        >>> flags = torch_nms(ltrb, scores, classes, thresh)
        >>> keep = np.nonzero(flags).view(-1)
        >>> ltrb[flags]
        >>> ltrb[keep]

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import torch
        >>> import numpy as np
        >>> # Test to check that conflicts are correctly resolved
        >>> ltrb = torch.FloatTensor(np.array([
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32))
        >>> scores = torch.FloatTensor(np.linspace(.8, .9, len(ltrb)))
        >>> classes = None
        >>> thresh = .3
        >>> keep = torch_nms(ltrb, scores, classes, thresh, fast=False)
        >>> bboxes[keep]
    """
    if ltrb.numel() == 0:
        return []

    # Sort coordinates by descending score
    ordered_scores, order = scores.sort(0, descending=True)

    import kwimage

    boxes = kwimage.Boxes(ltrb[order], 'ltrb')
    ious = boxes.ious(boxes, bias=bias)

    # if False:
    #     x1, y1, x2, y2 = ltrb[order].split(1, 1)

    #     # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
    #     dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp_(min=0)
    #     dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp_(min=0)

    #     # Compute iou
    #     intersections = dx * dy
    #     areas = (x2 - x1) * (y2 - y1)
    #     unions = (areas + areas.t()) - intersections
    #     ious = intersections / unions

    # Filter based on iou (and class)
    # NOTE: We are using following convention:
    #     * suppress if overlap > thresh
    #     * consider if overlap <= thresh
    # This convention has the property that when thresh=0, we dont just
    # remove everything.
    if _TORCH_HAS_BOOL_COMP:
        conflicting = (ious > thresh).byte().triu(1).bool()
    else:
        # Old way
        conflicting = (ious > thresh).triu(1)

    if classes is not None:
        ordered_classes = classes[order]
        same_class = (ordered_classes.unsqueeze(0) == ordered_classes.unsqueeze(1))
        conflicting = (conflicting & same_class)
    # Now we have a 2D matrix where conflicting[i, j] indicates if box[i]
    # conflicts with box[j]. For each box[i] we want to only keep the first
    # one that does not conflict with any other box[j].

    # Find out how many conflicts each ordered box has with other boxes that
    # have higher scores than it does. In other words...
    # n_conflicts[i] is the number of conflicts box[i] has with other boxes
    # that have a **higher score** than box[i] does. We will definately
    # keep any box where n_conflicts is 0, but we need to postprocess because
    # we might actually keep some boxes currently marked as conflicted.
    n_conflicts = conflicting.sum(0).byte()

    if not fast:
        # It is not enought to simply use all places where there are no
        # conflicts. Say we have boxes A, B, and C, where A conflicts with B,
        # B conflicts with C but A does not conflict with C. The fact that we
        # use A should mean that C is not longer conflicted.

        if True:
            # Marginally faster. best=618.2 us
            ordered_keep = np.zeros(len(conflicting), dtype=np.uint8)
            supress = np.zeros(len(conflicting), dtype=np.bool)
            for i, row in enumerate(conflicting.cpu().numpy() > 0):
                if not supress[i]:
                    ordered_keep[i] = 1
                    supress[row] = 1
            ordered_keep = torch.ByteTensor(ordered_keep).to(ltrb.device)
        else:
            # Marginally slower: best=1.382 ms,
            n_conflicts_post = n_conflicts.cpu()
            conflicting = conflicting.cpu()

            keep_len = len(n_conflicts_post) - 1
            for i in range(1, keep_len):
                if n_conflicts_post[i] > 0:
                    n_conflicts_post -= conflicting[i]

            n_conflicts = n_conflicts_post.to(n_conflicts.device)
            ordered_keep = (n_conflicts == 0)
    else:
        # Now we can simply keep any box that has no conflicts.
        ordered_keep = (n_conflicts == 0)

    # Unsort, so keep is aligned with input boxes
    shape = ordered_keep.size()
    keep = ordered_keep.new(*shape)
    keep.scatter_(0, order, ordered_keep)
    return keep


def test_class_torch():
    import numpy as np
    import torch
    import ubelt as ub
    import kwarray
    import kwimage

    thresh = .5

    num = 500
    rng = kwarray.ensure_rng(0)
    cpu_boxes = kwimage.Boxes.random(num, scale=400.0, rng=rng, format='ltrb', tensor=True)
    cpu_ltrb = cpu_boxes.to_ltrb().data
    # cpu_scores = torch.Tensor(rng.rand(len(cpu_ltrb)))
    # make all scores unique to ensure comparability
    cpu_scores = torch.Tensor(np.linspace(0, 1, len(cpu_ltrb)))
    cpu_cls = torch.LongTensor(rng.randint(0, 10, len(cpu_ltrb)))

    ltrb = cpu_boxes.to_ltrb().data.to('cuda')
    scores = cpu_scores.to('cuda')
    classes = cpu_cls.to('cuda')

    keep1 = []
    for idxs in ub.group_items(range(len(classes)), classes.cpu().numpy()).values():
        # cls_ltrb = ltrb.take(idxs, axis=0)
        # cls_scores = scores.take(idxs, axis=0)
        cls_ltrb = ltrb[idxs]
        cls_scores = scores[idxs]
        cls_keep = torch_nms(cls_ltrb, cls_scores, thresh=thresh, bias=0)
        keep1.extend(list(ub.compress(idxs, cls_keep.cpu().numpy())))
    keep1 = sorted(keep1)

    keep_ = torch_nms(ltrb, scores, classes=classes, thresh=thresh, bias=0)
    keep2 = np.where(keep_.cpu().numpy())[0].tolist()

    keep3 = kwimage.non_max_supression(ltrb.cpu().numpy(),
                                       scores.cpu().numpy(),
                                       classes=classes.cpu().numpy(),
                                       thresh=thresh, bias=0, impl='gpu')

    print(len(keep1))
    print(len(keep2))
    print(len(keep3))

    print(set(keep1) - set(keep2))
    print(set(keep2) - set(keep1))


if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
