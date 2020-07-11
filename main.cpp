
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <intrin.h>
#include "sgemm_avx256.h"

#include <Windows.h>

// "SAME" zero padding 
// stride : 1 only
// batch size : 1 only
void NCHW_convolution_naive_f32(
  size_t padded_feature_width,
  size_t padded_feature_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  size_t output_channels,
  const float* input,
  const float* weight,
  const float* bias,
  float* output
  )
{
  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t feature_width = padded_feature_width - padding_width;
  size_t feature_height = padded_feature_height - padding_height;
  for (size_t h=0; h<feature_height; ++h) {
    for (size_t w=0; w<feature_width; ++w) {
      for (size_t m=0; m<output_channels; ++m) {
        float sum = bias[m];
        for (size_t d=0; d<input_channels; ++d) {
          for (size_t y=0; y<kernel_height; ++y) {
            for (size_t x=0; x<kernel_width; ++x) {
              // output[m][h][w] += input[d][h+y][w+x] * weight[m][d][y][x];
              size_t input_index = d * padded_feature_height * padded_feature_width
                                   + padded_feature_width * (h + y) + (w + x);
              size_t weight_index = m * input_channels * kernel_height * kernel_width
                                    + d * kernel_height * kernel_width
                                    + kernel_width * y + x;
              sum += input[input_index] * weight[weight_index];
            }
          }
        }
        size_t output_index = m * padded_feature_height * padded_feature_width
                              + padded_feature_width * h + w;
        output[output_index] = sum;
      }
    }
  }
}

// "SAME" zero padding 
// stride : 1 only
// batch size : 1 only
void NHWC_convolution_naive_f32(
  size_t padded_feature_width,
  size_t padded_feature_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  size_t output_channels,
  const float* input,
  const float* weight,
  const float* bias,
  float* output
  )
{
  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t feature_width = padded_feature_width - padding_width;
  size_t feature_height = padded_feature_height - padding_height;
  for (size_t h=0; h<feature_height; ++h) {
    for (size_t w=0; w<feature_width; ++w) {
      for (size_t m=0; m<output_channels; ++m) {
        float sum = bias[m];
        for (size_t y=0; y<kernel_height; ++y) {
          for (size_t x=0; x<kernel_width; ++x) {
            for (size_t d=0; d<input_channels; ++d) {
              // output[h][w][m] += input[h+y][w+x][d] * weight[m][y][x][d];
              size_t input_index = (padded_feature_width * (h + y) + (w + x)) * input_channels + d;
              size_t weight_index = (m * kernel_height * kernel_width + (kernel_width * y + x)) * input_channels + d;
              sum += input[input_index] * weight[weight_index];
            }
          }
        }
        size_t output_index = (padded_feature_width * h + w) * output_channels + m;
        output[output_index] = sum;
      }
    }
  }
}

void NCHW_im2col(
  size_t feature_width,
  size_t feature_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  const float* im,
  float* col
  )
{
  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t padded_feature_width = feature_width + padding_width;
  size_t padded_feature_height = feature_height + padding_height;

  size_t col_index = 0;
  for (size_t h=0; h<feature_height; ++h) {
    for (size_t w=0; w<feature_width; ++w) {
      for (size_t d=0; d<input_channels; ++d) {
        for (size_t y=0; y<kernel_height; ++y) {
          for (size_t x=0; x<kernel_width; ++x) {
            size_t im_index = d * padded_feature_height * padded_feature_width
                              + (h + y) * padded_feature_width + (w + x);
            col[col_index] = im[im_index];
            ++col_index;
          }
        }
      }
    }
  }
}

void NHWC_im2col(
  size_t feature_width,
  size_t feature_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  const float* im,
  float* col
  )
{
  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t padded_feature_width = feature_width + padding_width;
  size_t padded_feature_height = feature_height + padding_height;

  size_t col_index = 0;
  for (size_t h=0; h<feature_height; ++h) {
    for (size_t w=0; w<feature_width; ++w) {
      for (size_t y=0; y<kernel_height; ++y) {
        for (size_t x=0; x<kernel_width; ++x) {
          for (size_t d=0; d<input_channels; ++d) {
            size_t im_index = ((h + y) * padded_feature_width + (w + x)) * input_channels + d;
            col[col_index] = im[im_index];
            ++col_index;
          }
        }
      }
    }
  }
}

void sgemm_ver1(
  const float* __restrict a,
  const float* __restrict b,
  float* __restrict c,
  size_t acols,
  size_t arows,
  size_t bcols
  )
{
  for (size_t i=0; i<arows; ++i) {
    for (size_t j=0; j<bcols; ++j) {
      for (size_t k=0; k<acols; ++k) {
        c[bcols * i + j] += a[acols * i + k] * b[bcols * k + j];
      }
    }
  }
}

void sgemm_ver3(
  const float* __restrict a,
  const float* __restrict b,
  float* __restrict c,
  size_t acols,
  size_t arows,
  size_t bcols
  )
{
  for (size_t i=0; i<arows; ++i) {
    for (size_t k=0; k<acols; ++k) {
      for (size_t j=0; j<bcols; ++j) {
        c[bcols * i + j] += a[acols * i + k] * b[bcols * k + j];
      }
    }
  }
}

void sgemm_ver4(
  const float* __restrict a,
  const float* __restrict b,
  float* __restrict c,
  size_t acols,
  size_t arows,
  size_t bcols
  )
{
  assert(bcols % 8 == 0);
  for (size_t i=0; i<arows; ++i) {
    for (size_t k=0; k<acols; ++k) {
      __m256 lik = _mm256_broadcast_ss(&a[i*acols+k]);
      for (size_t j=0; j<bcols; j+=8) {
        __m256 o = _mm256_loadu_ps(&c[i*acols+j]);
        __m256 r = _mm256_loadu_ps(&b[k*bcols+j]);
        o = _mm256_fmadd_ps(lik, r, o);
        _mm256_storeu_ps(&c[i*acols+j], o);
      }
    }
  }
}

void sgemm_ver7(
  const float* __restrict a,
  const float* __restrict b,
  float* __restrict c,
  size_t acols,
  size_t arows,
  size_t bcols
  )
{
}

int main(int argc, char* argv[])
{
  size_t feature_width = 64;
  size_t feature_height = 64;
  size_t kernel_width = 3;
  size_t kernel_height = 3;
  size_t input_channels = 64;
  size_t output_channels = 256;

  size_t flops = feature_height * feature_width * output_channels * input_channels * kernel_height * kernel_width * 2;
  printf("feature_width : %zu\n", feature_width);
  printf("feature_height : %zu\n", feature_height);
  printf("kernel_width : %zu\n", kernel_width);
  printf("kernel_height : %zu\n", kernel_height);
  printf("input_channels : %zu\n", input_channels);
  printf("output_channels : %zu\n", output_channels);
  printf("giga floating point operations : %.1f\n", flops/(double)(1000*1000*1000));

  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t padded_feature_width = feature_width + padding_width;
  size_t padded_feature_height = feature_height + padding_height;
  size_t input_bytes = padded_feature_height * padded_feature_width * input_channels * sizeof(float);
  size_t col_bytes = (kernel_width * kernel_height * input_channels) * (feature_width * feature_height) * sizeof(float);
  size_t weight_bytes = output_channels * kernel_height * kernel_width * input_channels * sizeof(float);
  size_t bias_bytes = output_channels * sizeof(float);
  size_t output_bytes = padded_feature_height * padded_feature_width * output_channels * sizeof(float);
  float* input = (float*)malloc(input_bytes);
  float* col = (float*)malloc(col_bytes);
  float* weight = (float*)malloc(weight_bytes);
  float* bias = (float*)malloc(bias_bytes);
  float* output = (float*)malloc(output_bytes);
  memset(input, 0, input_bytes);
  memset(col, 0, col_bytes);
  memset(output, 0, output_bytes);

  LARGE_INTEGER start, stop, freq;

  QueryPerformanceFrequency(&freq);

#define MEASURE_START QueryPerformanceCounter(&start)
#define MEASURE_STOP QueryPerformanceCounter(&stop)
#define MEASURE_REPORT(name) printf(#name " : %.1f msec\n", 1000 * (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart)
  
  MEASURE_START;
  NCHW_convolution_naive_f32(
    padded_feature_width,
    padded_feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    output_channels,
    input,
    weight,
    bias,
    output + padded_feature_width + 1
  );
  MEASURE_STOP;
  MEASURE_REPORT(NCHW_convolution_naive_f32);

  MEASURE_START;
  NHWC_convolution_naive_f32(
    padded_feature_width,
    padded_feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    output_channels,
    input,
    weight,
    bias,
    output + padded_feature_width + 1
  );
  MEASURE_STOP;
  MEASURE_REPORT(NHWC_convolution_naive_f32);
  
  MEASURE_START;
  NCHW_im2col(
    feature_width,
    feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    input,
    col
  );
  MEASURE_STOP;
  MEASURE_REPORT(NCHW_im2col);

  MEASURE_START;
  NHWC_im2col(
    feature_width,
    feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    input,
    col
  );
  MEASURE_STOP;
  MEASURE_REPORT(NHWC_im2col);

  size_t acols = (kernel_width * kernel_height * input_channels);
  size_t arows = output_channels;
  size_t bcols = (feature_width * feature_height);
  assert(flops == acols * arows * bcols * 2);

  //MEASURE_START
  //sgemm_ver1(
  //  weight,
  //  col,
  //  output,
  //  acols,
  //  arows,
  //  bcols
  //);
  //MEASURE_STOP
  //MEASURE_REPORT(sgemm_ver1)

  MEASURE_START;
  sgemm_ver3(
    weight,
    col,
    output,
    acols,
    arows,
    bcols
  );
  MEASURE_STOP;
  MEASURE_REPORT(sgemm_ver3);

  MEASURE_START;
  sgemm_ver4(
    weight,
    col,
    output,
    acols,
    arows,
    bcols
  );
  MEASURE_STOP;
  MEASURE_REPORT(sgemm_ver4);

  MEASURE_START;
  sgemm_ver7(
    weight,
    col,
    output,
    acols,
    arows,
    bcols
  );
  MEASURE_STOP;
  MEASURE_REPORT(sgemm_ver7);

  MEASURE_START;
  avx256_noncblas_sgemm(
    arows,  // M
    bcols,  // N
    acols,  // K
    1.0f,   // alpha
    weight, // A
    arows,  // lda
    col,    // B
    acols,  // ldb
    1.0f,   // beta
    output, // C
    arows   // ldc
  );
  MEASURE_STOP;
  MEASURE_REPORT(avx256_noncblas_sgemm);

  free(input);
  free(col);
  free(weight);
  free(bias);
  free(output);

  return 0;
}

