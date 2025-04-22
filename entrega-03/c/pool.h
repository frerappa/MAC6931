#pragma once

#include "definitions.h"

void maxPoolLayer(float features[FILTERS][IMG_ROWS][IMG_COLS], float pooledFeatures[FILTERS][POOL_IMG_ROWS][POOL_IMG_COLS]);
