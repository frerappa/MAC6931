#include "cnn.h"
#include "utils.h"
#include "activ_fun.h"
#include "conv.h"
#include "pool.h"
#include "flat.h"
#include "dense.h"

#include <stdio.h>

void cnn(float img_in[IMG_ROWS][IMG_COLS], float prediction[DIGITS]) {
  float pad_img[PAD_IMG_ROWS][PAD_IMG_COLS] = { 0 };
  normalizationAndPadding(img_in, pad_img);

  float features[FILTERS][IMG_ROWS][IMG_COLS] = { 0 };
  convolutionalLayer(pad_img, features);

  float pool_features[FILTERS][POOL_IMG_ROWS][POOL_IMG_COLS] = { 0 };
  maxPoolLayer(features, pool_features);

  float flat_array[FLAT_SIZE] = { 0 };
  flattenLayer(pool_features, flat_array);

  denseLayer(flat_array, prediction);
}
