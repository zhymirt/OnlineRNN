//
// Created by zhymi on 8/30/2023.
//

#ifndef ONLINERNN_TEST_FORECASTER_H
#define ONLINERNN_TEST_FORECASTER_H

#include <iostream> // cout, cin, endl, etc.
#include "forecaster.h"
#include "TorchRNN.h"
#include "CustomModels/RecurrentNNTorch.h"

using std::cout;
using std::endl;

typedef std::shared_ptr<TorchModels::TorchRNN> sharedModule;

int testUpdateWeightsRNNToRNN(sharedModule model1, sharedModule model2);

#endif //ONLINERNN_TEST_FORECASTER_H
