#include "cnn.hh"
#include "utils.hh"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#define N 10000

int readImages(const char *file, float ****images) {
  FILE *fp = fopen(file, "r");
  if (fp == NULL) 
    return -1;

  fseek(fp, 0, SEEK_SET); 

  *images = (float ***) malloc(N * sizeof(float **));
  for (int i = 0; i < N; i++) {
    (*images)[i] = (float **) malloc(IMG_ROWS * sizeof(float *));
    for (int j = 0; j < IMG_ROWS; j++) {
      (*images)[i][j] = (float *) malloc(IMG_COLS * sizeof(float));
    }
  }

  for (int i = 0; i < N; i++) {
    for (int x = 0; x < IMG_ROWS; x++) {
      for (int y = 0; y < IMG_COLS; y++) {
        fscanf(fp, "%f", &(*images)[i][x][y]);
      }
    }
  }

  return fclose(fp);
}

int readLabels(const char *file, int **labels) {
  FILE *fp = fopen(file, "r");
  if (fp == NULL) return -1;

  *labels = (int *) malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    fscanf(fp, "%d", &(*labels)[i]);
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

int main() {
  if ((0 == (KRN_ROWS % 2)) || (0 == (KRN_COLS % 2))) {
    printf("Error: odd kernel sizes are mandatory for this implementation\n");
    return 1;
  }

  float ***images = NULL;
  int *labels = NULL;
  double* latencies = (double*) malloc(N * sizeof(double)); 

  if (0 != readImages("data/in.dat", &images)) {
    printf("Error: can't open file ``data/in.dat''\n");
    return 1;
  }

  if (0 != readLabels("data/out.dat", &labels)) {
    printf("Error: can't open file ``data/out.dat''\n");
    return 1;
  }

  double time = 0;
  int correctPredictions = 0;
  float prediction[DIGITS];

  for (int i = 0; i < N; i++) {
    float img_local[IMG_ROWS][IMG_COLS];
    float img_linear[IMG_ROWS * IMG_COLS];

    for (int r = 0; r < IMG_ROWS; r++){
      for (int c = 0; c < IMG_COLS; c++) {
        img_local[r][c] = images[i][r][c];
        img_linear[r * IMG_COLS + c] = images[i][r][c];
      }
    }

    clock_t begin = clock();
    cnn(img_linear, prediction);
    clock_t end = clock();

    int currentPrediction = argmax(prediction);
    if (currentPrediction == labels[i]) {
      correctPredictions++;
    } 
    else {
      printf("\nExpected: %d\n", labels[i]);
      float pad_img[PAD_IMG_ROWS][PAD_IMG_COLS];
      normalizationAndPadding(img_local, pad_img);
      print_pad_img(pad_img);
      printf("Prediction:\n");
      for (int j = 0; j < DIGITS; j++) {
        printf("%d: %f\n", j, prediction[j]);
      }
      printf("\n");
    }

    double timeElapsed = (double)(end - begin) / CLOCKS_PER_SEC;
    time += timeElapsed;
    latencies[i] = timeElapsed;
  }

  double mean = time / N;
  double correctPredictionRate = correctPredictions * 100.0 / (double)N;

  double sumSqDiff = 0;
  for (int i = 0; i < N; i++) {
      double diff = latencies[i] - mean;
      sumSqDiff += diff * diff;
  }
  double stddev = sqrt(sumSqDiff / N);

  printf("\nCorrect predictions: %.2f %% (%d) total\n", correctPredictionRate, N);
  printf("Average Inference Time: %f ms | Standard Deviation: %f ms\n", 1000 * mean, 1000 * stddev );
  printf("Total Inference Time: %f s\n", time );
  printf("\n");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < IMG_ROWS; j++) {
      free(images[i][j]);
    }
    free(images[i]);
  }
  free(images);
  free(labels);
  free(latencies);

  return 0;
}
