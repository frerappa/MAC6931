#pragma once

#include "definitions.h"

void normalizationAndPadding(float img_in[IMG_ROWS][IMG_COLS], float img_out[PAD_IMG_ROWS][PAD_IMG_COLS]);

#ifndef __SYNTHESIS__
void print_img(float img[IMG_ROWS][IMG_COLS]);
void print_pad_img(float img[PAD_IMG_ROWS][PAD_IMG_COLS]);
void print_features(float features[FILTERS][IMG_ROWS][IMG_COLS]);
void print_pool_features(float pool_features[FILTERS][POOL_IMG_ROWS][POOL_IMG_COLS]);
#endif
