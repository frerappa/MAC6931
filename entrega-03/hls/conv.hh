#pragma once

#include "../headers/definitions.h"

#include "hls_stream.h"

void convolution(float padImg[PAD_IMG_ROWS][PAD_IMG_COLS], int filter, hls::stream<float>& convToPoolStream) ;

void convolutionalLayer(
    float padImg0[PAD_IMG_ROWS][PAD_IMG_COLS],
    float padImg1[PAD_IMG_ROWS][PAD_IMG_COLS],
    float padImg2[PAD_IMG_ROWS][PAD_IMG_COLS],
    float padImg3[PAD_IMG_ROWS][PAD_IMG_COLS],
    hls::stream<float> convToPoolStreams[FILTERS]);
