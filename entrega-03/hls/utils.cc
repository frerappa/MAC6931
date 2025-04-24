#include "../headers/definitions.h"
#include "utils.hh"

#ifndef __SYNTHESIS__
#include <cstdio>
#endif

void normalizationAndPadding(float imgIn[IMG_ROWS][IMG_COLS], float imgOut[PAD_IMG_ROWS][PAD_IMG_COLS]) {
  for (int row = 0; row < IMG_ROWS; row++) {
    for (int col = 0; col < IMG_COLS; col++) {
      imgOut[row + PAD_ROWS / 2][col + PAD_COLS / 2] = imgIn[row][col] / 255.0;
    }
  }
}

#ifndef __SYNTHESIS__

void print_pad_img(float img[PAD_IMG_ROWS][PAD_IMG_COLS]) {
  for (int i = 0; i < PAD_IMG_ROWS; i++) {
    for (int j = 0; j < PAD_IMG_COLS; j++) {
      printf("%.0f", img[i][j]);
    }
    printf("\n");
  }
}

void print_features(hls::stream<float> convToPoolStreams[FILTERS]) {
  for (int f = 0; f < FILTERS; f++) {
    printf("Feature map %d:\n", f);

    for (int row = 0; row < IMG_ROWS; row++) {
      for (int col = 0; col < IMG_COLS; col++) {
        printf("%.0f", convToPoolStreams[f].read());
      }
      printf("\n");
    }
  }
}

void print_pool_features(hls::stream<float> poolToFlatStreams[FILTERS]) {
  for (int f = 0; f < FILTERS; f++) {
    printf("Pool feature map %d:\n", f);

    for (int row = 0; row < POOL_IMG_ROWS; row++) {
      for (int col = 0; col < POOL_IMG_COLS; col++) {
        printf("%.0f", poolToFlatStreams[f].read());
      }
      printf("\n");
    }
  }
}

#endif
