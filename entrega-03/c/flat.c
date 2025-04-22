#include "flat.h"

void flattenLayer(float pooledFeatures[FILTERS][POOL_IMG_ROWS][POOL_IMG_COLS], float flatArray[FLAT_SIZE]) {
  int index = 0;

  for (int f = 0; f < FILTERS; f++) {
    for (int r = 0; r < POOL_IMG_ROWS; r++) {
      for (int c = 0; c < POOL_IMG_COLS; c++) {
        flatArray[index] = pooledFeatures[f][r][c];
        index++;
      }
    }
  }
}
