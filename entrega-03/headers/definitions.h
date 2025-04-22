#pragma once
#define DIGITS 10
#define IMG_ROWS 28
#define IMG_COLS 28

// Padding
#define KRN_ROWS 7
#define KRN_COLS 7
#define FILTERS 4
#define PAD_ROWS (KRN_ROWS - 1)
#define PAD_COLS (KRN_COLS - 1)
#define PAD_IMG_ROWS (IMG_ROWS + PAD_ROWS)
#define PAD_IMG_COLS (IMG_COLS + PAD_COLS)

// Pool layer
#define POOL_ROWS 2
#define POOL_COLS 2
#define POOL_IMG_ROWS (IMG_ROWS / POOL_ROWS)
#define POOL_IMG_COLS (IMG_COLS / POOL_COLS)

// Flatten layer
#define FLAT_SIZE (FILTERS * POOL_IMG_ROWS * POOL_IMG_COLS)

// Dense layer
#define DENSE_SIZE 10
