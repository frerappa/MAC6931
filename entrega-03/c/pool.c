#include "pool.h"

#include <float.h>

#pragma GCC diagnostic ignored "-Wunused-label"

void maxPool(
    float feature     [IMG_ROWS][IMG_COLS],
    float pooledFeature[POOL_IMG_ROWS][POOL_IMG_COLS]) {
  float maxSoFar = 0.0;

  for (int r = 0; r < IMG_ROWS; r += POOL_ROWS) {
    for (int c = 0; c < IMG_COLS; c += POOL_COLS) {
      maxSoFar = FLT_MIN;

      for (int poolRow = 0; poolRow < POOL_ROWS; poolRow++) {
        for (int poolColumn = 0; poolColumn < POOL_COLS; poolColumn++) {
          if (feature[r + poolRow][c + poolColumn] > maxSoFar) {
            maxSoFar = feature[r + poolRow][c + poolColumn];
          }
        }        
      }
      pooledFeature[r / POOL_ROWS][c / POOL_COLS] = maxSoFar;
    }
  }
}

void maxPoolLayer(
    float features     [FILTERS][IMG_ROWS][IMG_COLS],
    float pooledFeatures[FILTERS][POOL_IMG_ROWS][POOL_IMG_COLS]) {
  for (int f = 0; f < FILTERS; f++) {
    maxPool(features[f], pooledFeatures[f]);
  }
}
