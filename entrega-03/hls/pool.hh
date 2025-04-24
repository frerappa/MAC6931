#pragma once

#include "../headers/definitions.h"

#include "hls_stream.h"

void maxPoolLayer (hls::stream<float> conv_to_pool_streams[FILTERS], hls::stream<float> pool_to_flat_streams[FILTERS]);
