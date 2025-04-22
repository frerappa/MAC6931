#include "cnn.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include <math.h>

#define N 10000

int read_images(const char * file, float images[N][IMG_ROWS][IMG_COLS]) {
  FILE *fp;

  fp = fopen(file, "r");

  if (fp == NULL)
    return -1;

  for (int i = 0; i < N; i++)
    for (int x = 0; x < IMG_ROWS; ++x)
      for (int y = 0; y < IMG_COLS; ++y)
        if(fscanf(fp, "%f", & images[i][x][y]) != 1)
          return 1; 

  return fclose(fp);
}

int read_labels(const char * file, int labels[N]) {
  FILE *fp = fopen(file, "r");

  if (fp == NULL) {
    return -1;
  }

  for (int i = 0; i < N; i++) {
    if(fscanf(fp, "%d", & labels[i]) != 1) {
      return 1;

    }
  }

  return fclose(fp);
}

int argmax(float prediction[DIGITS]) {
  int maxIdx = 0;
  for (int i = 0; i < DIGITS; i++) {
    if (prediction[i] > prediction[maxIdx]) {
      maxIdx = i;
    }
  }
  return maxIdx;
}

int main () {

  float (*images)[IMG_ROWS][IMG_COLS] = malloc(sizeof(float[N][IMG_ROWS][IMG_COLS]));
  int *labels = malloc(sizeof(int[N]));
  double* latencies = (double*) malloc(N * sizeof(double)); 


  if (!images || !labels) {
      printf("Error: Failed to allocate memory.\n");
      return 1;
  }

  if (0 != read_images("../data/in.dat", images)) {
    printf("Error: can't open file ``../data/in.dat''\n");
    return 1;
  }

  if (0 != read_labels("../data/out.dat", labels)) {
    printf("Error: can't open file ``../data/out.dat''\n");
    return 1;
  }

  double time = 0;
  int correctPredictions = 0;
  float prediction[DIGITS];

  for (int i = 0; i < N; i++) {
    clock_t begin = clock();
    cnn(images[i], prediction);
    clock_t end = clock();

    int currentPrediction = argmax(prediction);
    if (currentPrediction == labels[i]) {
      correctPredictions++;
    }
    else {
      printf("\nExpected:  %d\nPredicted: %d\n", labels[i], currentPrediction);
      float pad_img[PAD_IMG_ROWS][PAD_IMG_COLS];
      normalizationAndPadding(images[i], pad_img);
      print_pad_img(pad_img);
      printf("Predictions\n");
      for (int j = 0; j < DIGITS; j++)
        printf("%d: %f\n", j, prediction[j]);
      printf("\n********************************\n");
    }

    double timeElapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    time += timeElapsed;
    latencies[i] = timeElapsed;

  }

  double correctPRedictionRate = 100 * ((double) correctPredictions / (double) N);
  double mean = time / N;

  double sumSqDiff = 0;
  for (int i = 0; i < N; i++) {
      double diff = latencies[i] - mean;
      sumSqDiff += diff * diff;
  }
  double stddev = sqrt(sumSqDiff / N);

  printf("\nCorrect predictions: %.2f %% (%d) total\n", correctPRedictionRate, N);
  printf("Average Inference Time: %f ms | Standard Deviation: %f ms\n", 1000 * mean, 1000 * stddev );
  printf("Total Inference Time: %f s\n", time );


  free(images);
  free(labels);
  free(latencies);

  return 0;
}
