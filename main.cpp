
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include <intrin.h>
#include "sgemm_avx256.h"
#include <cblas.h>
#include <random>

#include <Windows.h>

// NCHW layout
// "SAME" zero padding 
// stride : 1 only
// batch size : 1 only
void convolution_naive_f32(
  size_t input_width,
  size_t input_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  size_t output_channels,
  size_t input_line_stride,
  size_t input_image_stride,
  size_t output_line_stride,
  size_t output_image_stride,
  const float* input,
  const float* weight,
  const float* bias,
  float* output
  )
{
  size_t padding_width = (kernel_width - 1) / 2;
  size_t padding_height = (kernel_width - 1) / 2;
  for (size_t h=0; h<input_height; ++h) {
    for (size_t w=0; w<input_width; ++w) {
      for (size_t m=0; m<output_channels; ++m) {
        float sum = bias[m];
        for (size_t d=0; d<input_channels; ++d) {
          for (size_t y=0; y<kernel_height; ++y) {
            for (size_t x=0; x<kernel_width; ++x) {
              // output[m][h][w] += input[d][h+y][w+x] * weight[m][d][y][x];
#if 0
              int32_t iy = (int32_t)(h + y) - padding_height;
              int32_t ix = (int32_t)(w + x) - padding_width;
              if (iy < 0 || 0 < ix
                || iy >= input_height || ix >= input_width)
              {
                continue;
              }
#else
              int32_t iy = (int32_t)(h + y);
              int32_t ix = (int32_t)(w + x);
#endif
              size_t input_index = d * input_image_stride
                                   + input_line_stride * iy + ix;
              size_t weight_index = m * input_channels * kernel_height * kernel_width
                                    + d * kernel_height * kernel_width
                                    + kernel_width * y + x;
              sum += input[input_index] * weight[weight_index];
            }
          }
        }
        size_t output_index = m * output_image_stride
                              + output_line_stride * h + w;
        output[output_index] = sum;
      }
    }
  }
}

// NCHW layout
// stride : 1 only
// batch size : 1 only
void im2col(
  size_t input_width,
  size_t input_height,
  size_t kernel_width,
  size_t kernel_height,
  size_t input_channels,
  size_t input_line_stride,
  size_t input_image_stride,
  size_t col_line_stride,
  size_t col_image_stride,
  const float* im,
  float* col
  )
{
  //size_t padding_width = (kernel_width - 1) / 2;
  //size_t padding_height = (kernel_height - 1) / 2;
  for (size_t d=0; d<input_channels; ++d) {
    for (size_t y=0; y<kernel_height; ++y) {
      for (size_t x=0; x<kernel_width; ++x) {
        for (size_t h=0; h<input_height; ++h) {
          for (size_t w=0; w<input_width; ++w) {
            int32_t iy = (int32_t)(h + y);
            int32_t ix = (int32_t)(w + x);
            //iy -= padding_height;
            //ix -= padding_width;
            //if (iy < 0 || 0 < ix
            //  || iy >= input_height || ix >= input_width)
            //{
            //  continue;
            //}
            size_t im_index = d * input_image_stride
                              + iy * input_line_stride + ix;
            size_t col_index = (d * kernel_height * kernel_width + y * kernel_width + x) * col_image_stride
                               + col_line_stride * h + w;
            col[col_index] = im[im_index];
          }
        }
      }
    }
  }
}

void sgemm(
  const float* __restrict a,
  const float* __restrict b,
  float* __restrict c,
  size_t M,
  size_t N,
  size_t K
  )
{
  for (size_t m=0; m<M; ++m) {
    for (size_t k=0; k<K; ++k) {
      for (size_t n=0; n<N; ++n) {
        c[N * m + n] += a[K * m + k] * b[N * k + n];
      }
    }
  }
}

LARGE_INTEGER start, stop, freq;
size_t num_ops;
bool should_print_flops = true;

inline void measure_start()
{
  QueryPerformanceCounter(&start);
}
inline void measure_stop(const char* name)
{
  QueryPerformanceCounter(&stop);
  double elapsedSecond = (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart;
  double flops = num_ops / elapsedSecond;
  printf("%s : %.1f msec", name, 1000.0f * elapsedSecond);
  if (should_print_flops) {
    printf(" %.1f GFLOPS", flops/(1000*1000*1000));
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  size_t input_width = 128;
  size_t input_height = 128;
  size_t kernel_width = 3;
  size_t kernel_height = 3;
  size_t input_channels = 64;
  size_t output_channels = 256;

  num_ops = input_height * input_width * output_channels * input_channels * kernel_height * kernel_width * 2;
  printf("input_width : %zu\n", input_width);
  printf("input_height : %zu\n", input_height);
  printf("kernel_width : %zu\n", kernel_width);
  printf("kernel_height : %zu\n", kernel_height);
  printf("input_channels : %zu\n", input_channels);
  printf("output_channels : %zu\n", output_channels);
  printf("giga floating point operations : %.1f\n", num_ops/(double)(1000*1000*1000));

  size_t padded_input_width = input_width + kernel_width - 1;
  size_t padded_input_height = input_height + kernel_height - 1;
  size_t input_bytes = padded_input_height * padded_input_width * input_channels * sizeof(float);
  size_t col_bytes = (kernel_height * kernel_width * input_channels) * (padded_input_height * padded_input_width) * sizeof(float);
  size_t weight_bytes = output_channels * (kernel_height * kernel_width * input_channels) * sizeof(float);
  size_t bias_bytes = output_channels * sizeof(float);
  size_t output_bytes = padded_input_height * padded_input_width * output_channels * sizeof(float);
  float* input = (float*)malloc(input_bytes);
  float* col = (float*)malloc(col_bytes);
  float* weight = (float*)malloc(weight_bytes);
  float* bias = (float*)malloc(bias_bytes);
  float* output0 = (float*)malloc(output_bytes);
  float* output1 = (float*)malloc(output_bytes);
  float* output2 = (float*)malloc(output_bytes);
  memset(input, 0, input_bytes);
  memset(col, 0, col_bytes);
  memset(bias, 0, bias_bytes);
  memset(output0, 0, output_bytes);
  memset(output1, 0, output_bytes);
  memset(output2, 0, output_bytes);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);
  mt.seed();
  for (size_t i=0; i<input_channels; ++i) {
    for (size_t j=0; j<input_height; ++j) {
      for (size_t k=0; k<input_width; ++k) {
        size_t input_index = i * padded_input_height * padded_input_width
                             + (j + 1) * padded_input_width + (k + 1);
        input[input_index] = dist(mt);
      }
    }
  }
  for (size_t i=0; i<output_channels * (kernel_height * kernel_width * input_channels); ++i) {
    weight[i] = dist(mt);
  }

  QueryPerformanceFrequency(&freq);

  measure_start();
  convolution_naive_f32(
    input_width,
    input_height,
    kernel_width,
    kernel_height,
    input_channels,
    output_channels,
    padded_input_width,
    padded_input_height * padded_input_width,
    padded_input_width,
    padded_input_height * padded_input_width,
    input,
    weight,
    bias,
    output0 + padded_input_width + 1
  );
  measure_stop("convolution_naive_f32");

  should_print_flops = false;
  measure_start();
  im2col(
    input_width,
    input_height,
    kernel_width,
    kernel_height,
    input_channels,
    padded_input_width,
    padded_input_height * padded_input_width,
    padded_input_width,
    padded_input_height * padded_input_width,
    input,
    col + padded_input_width + 1
  );
  measure_stop("im2col");

  should_print_flops = true;

  size_t acols = (input_channels * kernel_height * kernel_width);
  size_t arows = output_channels;
  size_t bcols = (padded_input_height * padded_input_width);

  size_t m = arows;
  size_t n = bcols;
  size_t k = acols;

  measure_start();
  sgemm(
    weight,
    col,
    output1,
    m,
    n,
    k
  );
  measure_stop("sgemm_ver3");

  measure_start();
  avx256_noncblas_sgemm(
    m,      // M
    n,      // N
    k,      // K
    1.0f,   // alpha
    weight, // A
    k,      // lda
    col,    // B
    n,      // ldb
    0.0f,   // beta
    output1, // C
    n       // ldc
  );
  measure_stop("avx256_noncblas_sgemm");

  measure_start();
  cblas_sgemm(
    CblasRowMajor,    // Order
    CblasNoTrans,     // TransA
    CblasNoTrans,     // TransB
    m,      // M
    n,      // N
    k,      // K
    1.0f,   // alpha
    weight, // A
    k,      // lda
    col,    // B
    n,      // ldb
    0.0f,   // beta
    output1, // C
    n       // ldc
  );
  measure_stop("cblas_sgemm");
  
  float diff_sum = 0.0f;
  for (size_t i=0; i<m*n; ++i) {
    float diff = output0[i] - output1[i];
    diff_sum += fabs(diff);
  }
  printf("Diff: %.3f\n", diff_sum);

  free(input);
  free(col);
  free(weight);
  free(bias);
  free(output0);
  free(output1);
  free(output2);

  return 0;
}

