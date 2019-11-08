// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include "gpu_nms.hpp"
#include <vector>
#include <iostream>


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b, float bias) 
{
  /*
  """
  Compute intersection-over-union (IoU) between two boxes efficiently on a GPU device

  Args:
      a (array): top_left_x, top_left_y, bot_right_x, bot_right_y
      b (array): top_left_x, top_left_y, bot_right_x, bot_right_y
      bias (float) : either 0 or 1

  """
  */
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + bias, 0.f), height = max(bottom - top + bias, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + bias) * (a[3] - a[1] + bias);
  float Sb = (b[2] - b[0] + bias) * (b[3] - b[1] + bias);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, 
                           const float nms_overlap_thresh,
                           const float bias,
                           const float *dev_boxes, 
                           unsigned long long *dev_mask)
{
  /*
  """
  Runs overlap check on a subset of boxes. The results is a populated dev_mask,
  where the position [i, j] is True if the boxes overlap more than the
  threshold amount.
  """
  */
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  const int nCols = 4;  // number of columns is 4 (used to be 5 when score was in the data)

  __shared__ float block_boxes[threadsPerBlock * nCols];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * nCols + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * nCols + 0];
    block_boxes[threadIdx.x * nCols + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * nCols + 1];
    block_boxes[threadIdx.x * nCols + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * nCols + 2];
    block_boxes[threadIdx.x * nCols + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * nCols + 3];
    /*block_boxes[threadIdx.x * nCols + 4] =                             */
    /*    dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * nCols + 4];*/
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * nCols;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) 
    {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) 
    {
      //  NOTE: We are using following convention:
      //      * suppress if overlap > thresh
      //      * consider if overlap <= thresh
      //  This convention has the property that when thresh=0, we dont just
      //  remove everything, 
      if (devIoU(cur_box, block_boxes + i * nCols, bias) > nms_overlap_thresh) 
      {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) 
{
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

void _nms_cuda(int* keep_out,
        int* num_out,
        const float* boxes_host,
        int boxes_num,
        int boxes_dim,
        float nms_overlap_thresh, 
        float bias,
        int device_id) 
{
  /*
  """
  Main entry point

      Args:
          keep_out (outvar): preallocated array where the 
              first `num_out` items will be populated with indices to keep.

          num_out (outvar): returns the number of boxes to keep

          boxes_host (float*): pointer to Nx4 array of TLBR boxes. 
              These boxes must be ordered by descending score.
              (Note the scores are not pass in due to this)

          boxes_num (int): number of input boxes

          boxes_dim (int): set to 4

          nms_overlap_thresh (float): algo arg

          bias (float): algo arg

          device_id (int): GPU to use

  """
  */
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  // Allocate space and copy data onto the GPU
  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  // the mask indicates a 2D boolean overlap matrix, 
  // note that the implementation is using bitmasks, so one (unsigned long
  // long) integer represents an entire row of this matrix efficiently.
  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  // Run parallel computation
  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  bias,
                                  boxes_dev,
                                  mask_dev);

  // Copy the masks off the GPU
  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  // remv[i] is set to True if we are going to remove the i-th box
  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  // Unpack results of GPU computation vias the mask_dev / mask_host
  // Parse the bit-packed representation of mask and use the greedy 
  // algorithm to compute a final association. The indices of the 
  // kept boxes are written to `keep_out`.
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {

      // Indicate that we will keep box i, because nothing has removed it yet.
      keep_out[num_to_keep++] = i;

      // Now remove anything that box i intersected with.
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}
