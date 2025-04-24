#pragma once

#include "../headers/definitions.h"

#include "hls_stream.h"

void denseLayerSoftMax(hls::stream<float> denseToSoftmaxStreams[FILTERS], float prediction[DIGITS]);

void dense(hls::stream<float>& flatToDenseStream, int filter, hls::stream<float>& denseToSoftmaxStream);