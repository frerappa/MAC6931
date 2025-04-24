#include "dense.hh"
#include "../headers/dense_weights.h"

#include <cmath>
#include "hls_stream.h"

void denseLayerSoftMax(
  hls::stream<float> denseToSoftmaxStreams[FILTERS],
  float prediction[DIGITS]
) {
  float sum;
  float expSum = 0.0;

  dense_soft_max_for_dense_size:
  for (int d = 0; d < DENSE_SIZE; d++) {
    sum = dense_biases[d];

    dense_soft_max_for_filters:
    for (int f = 0; f < FILTERS; f++) {
      sum += denseToSoftmaxStreams[f].read();
    }

    expSum += (prediction[d] = expf(sum));
  }

  dense_soft_max_for_digits:
  for (int p = 0; p < DIGITS; p++) {
    prediction[p] /= expSum;
  }
}

void dense(hls::stream<float>& flatToDenseStream, int filter, hls::stream<float>& denseToSoftmaxStream) {
  float flatValue;
  float denseArray[DENSE_SIZE] = { 0 };

  for (int i = 0; i < FLAT_SIZE / FILTERS; i++) {
    flatValue = flatToDenseStream.read();

    for (int d = 0; d < DENSE_SIZE; d++) {
      int index = filter * (FLAT_SIZE / FILTERS) + i;
      denseArray[d] += dense_weights[index][d] * flatValue;
    }
  }

  for (int j = 0; j < DENSE_SIZE; j++) {
    denseToSoftmaxStream.write(denseArray[j]);
  }
}

void denseLayer(hls::stream<float> flatToDenseStreams[FILTERS], hls::stream<float> denseToSoftmaxStreams[FILTERS]) {
  dense(flatToDenseStreams[0], 0, denseToSoftmaxStreams[0]);
  dense(flatToDenseStreams[1], 1, denseToSoftmaxStreams[1]);
  dense(flatToDenseStreams[2], 2, denseToSoftmaxStreams[2]);
  dense(flatToDenseStreams[3], 3, denseToSoftmaxStreams[3]);
}
