//
// Created by zhymi on 9/25/2023.
//

#ifndef ONLINERNN_TESTTORCHRNN_H
#define ONLINERNN_TESTTORCHRNN_H

#pragma warning(push, 0)
#include <torch/torch.h>
#pragma  warning(pop)
#include "Models/TorchRNN.h"
#include "Models/TorchMLP.h"


int testModelEmptyConstructionSucceeds();
int testModelParameterConstructionSucceeds(int histLength, int numLayers, torch::Dtype dtype);


int testForwardSucceeds(int histLength, int numLayers, torch::Dtype dtype, int numLoops=10);
//int testPredictSucceeds(); // sanity check since this function is alias
//int testLossSucceeds();
int testRandomHiddenSucceeds(int histLength, int numLayers, torch::Dtype dtype);
int testMakeHiddenStateSucceeds();

#endif //ONLINERNN_TESTTORCHRNN_H
