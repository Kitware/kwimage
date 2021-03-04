def time_lightnet_nms(ti, cpu_boxes, gpu, ydata, outputs, thresh):
    # FIXME
    # Format boxes in lightnet format
    cpu_ln_boxes = torch.cat([cpu_boxes.to_cxywh().data, cpu_scores[:, None], cpu_cls.float()[:, None]], dim=-1)
    gpu_ln_boxes = cpu_ln_boxes.to(gpu)

    def _ln_output_to_keep(ln_output, ln_boxes):
        keep = []
        for row in ln_output:
            # Find the index that we kept
            idxs = np.where(np.all(np.isclose(ln_boxes, row), axis=1))[0]
            assert len(idxs) == 1
            keep.append(idxs[0])
        assert np.all(np.isclose(ln_boxes[keep], ln_output))
        return keep

    from lightnet.data.transform._postprocess import NonMaxSupression
    for timer in ti.reset('lightnet-slow(gpu)'):
        with timer:
            ln_output = NonMaxSupression._nms(gpu_ln_boxes, nms_thresh=thresh, class_nms=False, fast=False)
            torch.cuda.synchronize()
    # convert lightnet NMS output to keep for consistency
    keep = _ln_output_to_keep(ln_output, gpu_ln_boxes)
    ydata[ti.label].append(ti.min())
    outputs[ti.label] = sorted(keep)

    if False:
        for timer in ti.reset('lightnet-fast(gpu)'):
            with timer:
                ln_output = NonMaxSupression._nms(gpu_ln_boxes, nms_thresh=thresh, class_nms=False, fast=True)
                torch.cuda.synchronize()
        # convert lightnet NMS output to keep for consistency
        keep = _ln_output_to_keep(ln_output, gpu_ln_boxes)
        ydata[ti.label].append(ti.min())
        outputs[ti.label] = sorted(keep)


# def raw_benchamrk_nms():
#     """
#     python -m netharn.util.nms.torch_nms _benchmark --show

#     SeeAlso:
#         PJR Darknet NonMax supression
#         https://github.com/pjreddie/darknet/blob/master/src/box.c

#         Lightnet NMS
#         https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/data/transform/_postprocess.py#L116

#     """

#     N = 100
#     bestof = 10

#     ydata = ub.ddict(list)
#     # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]

#     # max number of boxes yolo will spit out at a time
#     max_boxes = 19 * 19 * 5

#     xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, max_boxes]
#     # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500]
#     # xdata = [10, 100, 500, 1000, 1500, 2000]
#     xdata = [10, 100, 500, 1000]

#     thresh = 0.5

#     for num in xdata:
#         print('\n\n---- number of boxes = {} ----\n'.format(num))

#         outputs = {}

#         ti = ub.Timerit(N, bestof=bestof)

#         # Build random test boxes and scores
#         np_dets = kwimage.Detections.random(num, scale=10.0, rng=0)
#         # make all scores unique to ensure comparability
#         np_dets.scores[:] = np.linspace(0, 1, np_dets.num_boxes())

#         gpu = torch.device('cuda', 0)

#         measure_gpu = torch.cuda.is_available()
#         measure_cpu = False or not torch.cuda.is_available()

#         # ----------------------------------

#         measure_daq = True
#         if measure_daq:
#             for timer in ti.reset('daq_cython(cpu)'):
#                 with timer:
#                     keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='cpu')
#                     torch.cuda.synchronize()
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#             for timer in ti.reset('daq_cython(gpu)'):
#                 with timer:
#                     keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='gpu')
#                     torch.cuda.synchronize()
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#             for timer in ti.reset('daq_cython(auto)'):
#                 with timer:
#                     keep = np_dets.non_max_supression(thresh=thresh, daq=True, impl='auto')
#                     torch.cuda.synchronize()
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#         if measure_cpu:
#             cpu_dets = np_dets.tensor(None)
#             cpu_ltrb = cpu_dets.boxes.to_ltrb().data
#             cpu_scores = cpu_dets.scores
#             for timer in ti.reset('torch(cpu)'):
#                 with timer:
#                     keep = torch_nms(cpu_ltrb, cpu_scores, thresh=thresh)
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#         if measure_gpu:
#             gpu_dets = np_dets.tensor(gpu)
#             # Move boxes to the GPU
#             gpu_ltrb = gpu_dets.boxes.to_ltrb().data
#             gpu_scores = gpu_dets.scores

#             for timer in ti.reset('torch(gpu)'):
#                 with timer:
#                     keep = torch_nms(gpu_ltrb, gpu_scores, thresh=thresh)
#                     torch.cuda.synchronize()
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#             np_ltrb = np_dets.boxes.to_ltrb().data
#             np_scores = np_dets.scores
#             for timer in ti.reset('cython(gpu)'):
#                 with timer:
#                     keep = kwimage.non_max_supression(np_ltrb, np_scores, thresh=thresh, impl='gpu')
#                     torch.cuda.synchronize()
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#         if True:
#             np_ltrb = np_dets.boxes.to_ltrb().data
#             np_scores = np_dets.scores
#             for timer in ti.reset('cython(cpu)'):
#                 with timer:
#                     keep = kwimage.non_max_supression(np_ltrb, np_scores, thresh=thresh, impl='cpu')
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#             for timer in ti.reset('numpy(cpu)'):
#                 with timer:
#                     keep = kwimage.non_max_supression(np_ltrb, np_scores, thresh=thresh, impl='py')
#             ydata[ti.label].append(ti.min())
#             outputs[ti.label] = ensure_numpy_indices(keep)

#         # ----------------------------------

#         # Check that all kept boxes do not have more than `threshold` ious
#         for key, keep_idxs in outputs.items():
#             kept = kwimage.Boxes(np_ltrb[keep_idxs], 'ltrb')
#             ious = kept.ious(kept)
#             max_iou = (np.tril(ious) - np.eye(len(ious))).max()
#             if max_iou > thresh:
#                 print('{} produced a bad result with max_iou={}'.format(key, max_iou))

#         # Check result consistency:
#         print('\nResult stats:')
#         for key in sorted(outputs.keys()):
#             print('    * {:<20}: num={}'.format(key, len(outputs[key])))

#         print('\nResult overlaps (method1, method2: jaccard):')
#         datas = []
#         for k1, k2 in it.combinations(sorted(outputs.keys()), 2):
#             idxs1 = set(outputs[k1])
#             idxs2 = set(outputs[k2])
#             jaccard = len(idxs1 & idxs2) / len(idxs1 | idxs2)
#             datas.append((k1, k2, jaccard))

#         datas = sorted(datas, key=lambda x: -x[2])
#         for k1, k2, jaccard in datas:
#             print('    * {:<20}, {:<20}: {:0.4f}'.format(k1, k2, jaccard))

#     kwimage.autompl()

#     ydata = ub.dict_subset(ydata, ub.argsort(ub.map_vals(lambda x: x[-1], ydata)))
#     kwimage.multi_plot(xdata, ydata, xlabel='num boxes', ylabel='seconds')
#     kwimage.show_if_requested()
