#include "activ_fun.h"

#include <math.h>

float relu(float x) {
  return x > 0.0 ? x : 0.0;     
}

void softMax(float denseArray[DIGITS], float predictions[DIGITS]) {
  float sum = 0.0;

  for (int i = 0; i < DIGITS; i++) {
    sum += expf(denseArray[i]);
  }

  for (int j = 0; j < DIGITS; j++)
  {
    predictions[j] = expf(denseArray[j]) / sum;
  }

}
