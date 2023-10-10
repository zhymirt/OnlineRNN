//
// Created by zhymi on 8/30/2023.
//

#ifndef ONLINERNN_TEST_FORECASTER_H
#define ONLINERNN_TEST_FORECASTER_H

#include <iostream> // cout, cin, endl, etc.
#include "forecaster.h"
#include "Models/TorchRNN.h"
//#include "CustomModels/RecurrentNNTorch.h"

using std::cout;
using std::endl;

typedef std::shared_ptr<TorchModels::TorchRNN> sharedModule;

int testUpdateWeightsRNNToRNN(sharedModule model1, sharedModule model2);
int testTorchRNNMakePredictionsTorchPipeSucceeds(int histLength, int numSamples, int batchSize);
//int testTorchMLPMakePredictionsTorchPipeSucceeds(int histLength, int numSamples, int batchSize)
//int testMakePredictionsTorchPipeSucceeds(sharedModule model, QueueType inQueue, QueueType outQueue, WeightShareType weightHolder);
int testTorchRNNMakeImprovementsTorchPipeSucceeds(int histLength, int numSamples, int batchSize);
int testMakeImprovementsTorchPipeSucceeds(sharedModule model, LearnQueueType in_queue, LossFunction loss_fn, int seq_length,
                                          OptimizerType optimizer, WeightShareType weightHolder, QueueType out_queue);

#endif //ONLINERNN_TEST_FORECASTER_H
