#pragma once

#include "definitions.h"

void convolutionalLayer(float paddedImage[PAD_IMG_ROWS][PAD_IMG_COLS], float features[FILTERS][IMG_ROWS][IMG_COLS]);
