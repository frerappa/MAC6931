#include "dense.h"
#include "dense_weights.h"
#include "activ_fun.h"

void denseLayer(float flatArray[FLAT_SIZE], float prediction[DIGITS]) {
  float weightedSum = 0.0;
  float denseArray[DENSE_SIZE] = { 0 };

  for (int d = 0; d < DENSE_SIZE; ++d) {
    weightedSum = 0.0;

    for (int f = 0; f < FLAT_SIZE; f++) {
      weightedSum += dense_weights[f][d] * flatArray[f];
    }

    denseArray[d] = weightedSum + dense_biases[d];
  }

  softMax(denseArray, prediction);
}

