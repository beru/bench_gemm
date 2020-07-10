
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
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

void sgemm(
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
      float sum = 0;
      for (size_t k=0; k<acols; ++k) {
        sum += a[acols * i + k] * b[bcols * k + j];
      }
      c[bcols * i + j] = sum;
    }
  }
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
  printf("feature_width : %d\n", feature_width);
  printf("feature_height : %d\n", feature_height);
  printf("kernel_width : %d\n", kernel_width);
  printf("kernel_height : %d\n", kernel_height);
  printf("input_channels : %d\n", input_channels);
  printf("output_channels : %d\n", output_channels);
  printf("GFLOPS : %.1f\n", flops/(double)(1000*1000*1000));

  size_t padding_width = kernel_width - 1;
  size_t padding_height = kernel_height - 1;
  size_t padded_feature_width = feature_width + padding_width;
  size_t padded_feature_height = feature_height + padding_height;
  size_t input_bytes = padded_feature_height * padded_feature_width * input_channels * sizeof(float);
  size_t col_bytes = (kernel_width * kernel_height * input_channels) * (feature_width * feature_height) * sizeof(float);
  size_t weight_bytes = output_channels * kernel_height * kernel_width * input_channels * sizeof(float);
  size_t bias_bytes = output_channels * sizeof(float);
  size_t output_bytes = padded_feature_height * padded_feature_width * output_channels * sizeof(float);
  float* input = malloc(input_bytes);
  float* col = malloc(col_bytes);
  float* weight = malloc(weight_bytes);
  float* bias = malloc(bias_bytes);
  float* output = malloc(output_bytes);
  memset(input, 0, input_bytes);
  memset(col, 0, col_bytes);
  memset(output, 0, output_bytes);

  LARGE_INTEGER start, stop, freq;
  LONGLONG elapsed;

  QueryPerformanceFrequency(&freq);

  QueryPerformanceCounter(&start);
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
  QueryPerformanceCounter(&stop);
  elapsed = stop.QuadPart - start.QuadPart;
  printf("NCHW_convolution_naive_f32 elapsed msec: %f\n", 1000 * (double)elapsed / freq.QuadPart);

  QueryPerformanceCounter(&start);
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
  QueryPerformanceCounter(&stop);
  elapsed = stop.QuadPart - start.QuadPart;
  printf("NHWC_convolution_naive_f32 elapsed msec: %f\n", 1000 * (double)elapsed / freq.QuadPart);

  QueryPerformanceCounter(&start);
  NCHW_im2col(
    feature_width,
    feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    input,
    col
  );
  QueryPerformanceCounter(&stop);
  elapsed = stop.QuadPart - start.QuadPart;
  printf("NCHW_im2col elapsed msec: %f\n", 1000 * (double)elapsed / freq.QuadPart);

  QueryPerformanceCounter(&start);
  NHWC_im2col(
    feature_width,
    feature_height,
    kernel_width,
    kernel_height,
    input_channels,
    input,
    col
  );
  QueryPerformanceCounter(&stop);
  elapsed = stop.QuadPart - start.QuadPart;
  printf("NHWC_im2col elapsed msec: %f\n", 1000 * (double)elapsed / freq.QuadPart);

  size_t acols = (kernel_width * kernel_height * input_channels);
  size_t arows = output_channels;
  size_t bcols = (feature_width * feature_height);
  assert(flops == acols * arows * bcols * 2);
  QueryPerformanceCounter(&start);
  sgemm(
    weight,
    col,
    output,
    acols,
    arows,
    bcols
  );
  QueryPerformanceCounter(&stop);
  elapsed = stop.QuadPart - start.QuadPart;
  printf("sgemm elapsed msec: %f\n", 1000 * (double)elapsed / freq.QuadPart);

  free(input);
  free(col);
  free(weight);
  free(bias);
  free(output);

  return 0;
}

