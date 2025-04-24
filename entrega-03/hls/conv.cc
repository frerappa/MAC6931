#include "conv.hh"
#include "../headers/conv_weights.h"
#include "../headers/definitions.h"

float relu(float x) {
  return x > 0.0 ? x : 0.0;     
}


void convolution(float paddedImage[PAD_IMG_ROWS][PAD_IMG_COLS], int filter, hls::stream<float>& convToPoolStream) {
  float weightedSum = 0.0;

  conv_for_rows:
  for (int row = 0; row < IMG_ROWS; row += POOL_ROWS) {
    conv_for_cols:
    for (int col = 0; col < IMG_COLS; col += POOL_COLS) {
      pool_for_rows:
      for (int poolRow = 0; poolRow < POOL_ROWS; poolRow++) {
        pool_for_cols:
        for (int poolCol = 0; poolCol < POOL_COLS; poolCol++) {
          weightedSum = 0.0;

          krn_for_rows:
          for (int kernelRow = 0; kernelRow < KRN_ROWS; ++kernelRow) {
            krn_for_cols:
            for (int kernelCol = 0; kernelCol < KRN_COLS; ++kernelCol) {
              float weight = conv_weights[filter][kernelRow][kernelCol];
              float pixel = paddedImage[row + poolRow + kernelRow][col + poolCol + kernelCol];
              weightedSum += weight * pixel;
            }
          }

          convToPoolStream.write(relu(weightedSum + conv_biases[filter]));
        }
      }
    }
  }
}

void convolutionalLayer(
    float paddedImage0[PAD_IMG_ROWS][PAD_IMG_COLS],
    float paddedImage1[PAD_IMG_ROWS][PAD_IMG_COLS],
    float paddedImage2[PAD_IMG_ROWS][PAD_IMG_COLS],
    float paddedImage3[PAD_IMG_ROWS][PAD_IMG_COLS],
    hls::stream<float> convToPoolStreams[FILTERS]) {
  convolution(paddedImage0, 0, convToPoolStreams[0]);
  convolution(paddedImage1, 1, convToPoolStreams[1]);
  convolution(paddedImage2, 2, convToPoolStreams[2]);
  convolution(paddedImage3, 3, convToPoolStreams[3]);
}

