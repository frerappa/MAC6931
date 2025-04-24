#pragma once

#include "../headers/definitions.h"

#include "hls_stream.h"

void flattenLayer(
    hls::stream<float> pool_to_flat_streams[FILTERS],
    hls::stream<float> flat_to_dense_stream[FILTERS]
);
