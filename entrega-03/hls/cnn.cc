#include "cnn.hh"
#include "utils.hh"
#include "conv.hh"
#include "pool.hh"
#include "flat.hh"
#include "dense.hh"

#include "hls_stream.h"

#ifndef __SYNTHESIS__
# include <cstdio>
#endif

void dataflowSection(
    float pad_img0[PAD_IMG_ROWS][PAD_IMG_COLS], float pad_img1[PAD_IMG_ROWS][PAD_IMG_COLS], 
    float pad_img2[PAD_IMG_ROWS][PAD_IMG_COLS], float pad_img3[PAD_IMG_ROWS][PAD_IMG_COLS], float prediction[DIGITS]) {

  hls::stream<float, IMG_ROWS * IMG_COLS>
  conv_to_pool_streams[FILTERS];

  convolutionalLayer(pad_img0, pad_img1, pad_img2, pad_img3,
                      conv_to_pool_streams);

  hls::stream<float, POOL_IMG_ROWS * POOL_IMG_COLS>
  pool_to_flat_streams[FILTERS];
  maxPoolLayer(conv_to_pool_streams, pool_to_flat_streams);

  hls::stream<float, FLAT_SIZE / FILTERS> flat_to_dense_streams[FILTERS];
  flattenLayer(pool_to_flat_streams, flat_to_dense_streams);
  hls::stream<float, DENSE_SIZE> dense_to_softmax_streams[FILTERS];
  denseLayer(flat_to_dense_streams, dense_to_softmax_streams);

  denseLayerSoftMax(dense_to_softmax_streams, prediction);
}

void cnn(float *img_in, float *prediction) {
  #pragma HLS INTERFACE m_axi port=img_in offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=prediction offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=img_in
  #pragma HLS INTERFACE s_axilite port=prediction
  #pragma HLS INTERFACE s_axilite port=return

  float img_local[IMG_ROWS][IMG_COLS];
  for (int i = 0; i < IMG_ROWS; i++) {
    for (int j = 0; j < IMG_COLS; j++) {
      img_local[i][j] = img_in[i * IMG_COLS + j];
    }
  }

  float pad_img0[PAD_IMG_ROWS][PAD_IMG_COLS] = { 0 };
  normalizationAndPadding(img_local, pad_img0);

  float pad_img1[PAD_IMG_ROWS][PAD_IMG_COLS];
  float pad_img2[PAD_IMG_ROWS][PAD_IMG_COLS];
  float pad_img3[PAD_IMG_ROWS][PAD_IMG_COLS];

  for (int i = 0; i < PAD_IMG_ROWS; i++) {
    for (int j = 0; j < PAD_IMG_COLS; j++) {
      pad_img1[i][j] = pad_img0[i][j];
      pad_img2[i][j] = pad_img0[i][j];
      pad_img3[i][j] = pad_img0[i][j];
    }
  }


  float local_prediction[DIGITS];
  dataflowSection(pad_img0, pad_img1, pad_img2, pad_img3, local_prediction);

  for (int i = 0; i < DIGITS; i++) {
    prediction[i] = local_prediction[i];
  }
}