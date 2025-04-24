############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_top -name cnn "cnn"
set_directive_dataflow "dataflowSection"
set_directive_dataflow "convolutionalLayer"
set_directive_dataflow "maxPoolLayer"
set_directive_dataflow "flattenLayer"
set_directive_dataflow "denseLayer"
set_directive_pipeline -II 32 "convolution/conv_for_cols"
set_directive_pipeline -II 4 "max_pooling/pool_for_cols"
set_directive_pipeline -II 10 "dense/dense_for_flat"
