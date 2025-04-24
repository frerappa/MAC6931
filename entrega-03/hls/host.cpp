#include "cnn.hh"
#include "utils.hh"
#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl2.hpp>
#define N 10000

int main(int argc, char **argv)
{
  std::vector<std::vector<float>> images(N, std::vector<float>(IMG_ROWS * IMG_COLS));
  std::vector<int> labels(N);

  std::ifstream fin_images("data/in.dat");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < IMG_ROWS * IMG_COLS; j++) {
      fin_images >> images[i][j];
    }
  }

  std::ifstream fin_labels("data/out.dat");
  for (int i = 0; i < N; i++) {
    fin_labels >> labels[i];
  }
  // cl::Platform platform =
  // cl::Device device =
  // cl::Context context(device);
  // cl::CommandQueue queue(context, device);

  std::ifstream bin("cnn_kernel.xclbin", std::ios::binary);
  std::vector<unsigned char> bin_data((std::istreambuf_iterator<char>(bin)), {});
  cl::Program program(context, {device}, {bin_data});
  cl::Kernel kernel(program, "cnn");

  cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, sizeof(float) * N * IMG_ROWS * IMG_COLS);
  cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * DIGITS);

  queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, sizeof(float) * N * IMG_ROWS * IMG_COLS, images.data());

  kernel.setArg(0, buffer_input);
  kernel.setArg(1, buffer_output);

  queue.enqueueTask(kernel);
  queue.finish();

  std::vector<float> prediction(DIGITS);
  queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, sizeof(float) * DIGITS, prediction.data());

  int correct_predictions = 0;
  for (int i = 0; i < N; i++) {
    int predicted_label = get_max_prediction(prediction);
    if (predicted_label == labels[i])
      ++correct_predictions;
  }

  double correct_predictions_perc = correct_predictions * 100.0 / N;
  std::cout << "Total predictions: " << N << std::endl;
  std::cout << "Correct predictions: " << correct_predictions_perc << " %" << std::endl;

  return 0;
}
