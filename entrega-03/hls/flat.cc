#include "flat.hh"

void flattening(hls::stream<float> &pool_to_flat_stream, hls::stream<float> &flat_to_dense_stream) {
  flat_for_rows:
  for (int r = 0; r < POOL_IMG_ROWS; r++) {
    flat_for_cols:
    for (int c = 0; c < POOL_IMG_COLS; c++) {
      flat_to_dense_stream.write(pool_to_flat_stream.read());
    }
  }
}

void flattenLayer (hls::stream<float> poolToFlatStreams[FILTERS], hls::stream<float> flatToDenseStreams[FILTERS]) {
  flattening(poolToFlatStreams[0], flatToDenseStreams[0]);
  flattening(poolToFlatStreams[1], flatToDenseStreams[1]);
  flattening(poolToFlatStreams[2], flatToDenseStreams[2]);
  flattening(poolToFlatStreams[3], flatToDenseStreams[3]);
}
