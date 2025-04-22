#include "conv.h"
#include "definitions.h"
#include "activ_fun.h"
#include "conv_weights.h"

void convolutionalLayer(float paddedImage[PAD_IMG_ROWS][PAD_IMG_COLS], float features[FILTERS][IMG_ROWS][IMG_COLS]) {
  for (int f = 0; f < FILTERS; f++) {
    float weightedSum = 0.0; 

    for (int r = 0; r < IMG_ROWS; r++) {
      for (int c = 0; c < IMG_COLS; c++) {
          weightedSum = 0.0;

          for (int kr = 0; kr < KRN_ROWS; kr++) {
            for (int kc = 0; kc < KRN_COLS; kc++) { 
              weightedSum += conv_weights[f][kr][kc] * paddedImage[r + kr][c + kc];
            }
          }
          features[f][r][c] = relu(weightedSum + conv_biases[f]);
        }
    }
  }
}
