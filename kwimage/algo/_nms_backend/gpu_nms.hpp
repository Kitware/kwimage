void _nms_cuda(int* keep_out, int* num_out, const float* boxes_host, int
    boxes_num, int boxes_dim, float nms_overlap_thresh, float bias, int
    device_id);
