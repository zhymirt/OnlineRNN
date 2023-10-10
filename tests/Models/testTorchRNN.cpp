//
// Created by zhymi on 9/25/2023.
//

#include "testTorchRNN.h"

int testModelEmptyConstructionSucceeds() {
    TorchModels::TorchRNN model;
    return 0;
}

int testModelParameterConstructionSucceeds(int histLength, int numLayers, torch::Dtype dtype) {
    TorchModels::TorchRNN model(histLength, numLayers, dtype);
    if (model.getInputLength() == histLength && model.getNumLayers() == numLayers && model.getDtype() == dtype)
        return 0;
    return 1;
}

int testForwardSucceeds(int histLength, int numLayers, torch::Dtype dtype, int numLoops) {
    torch::NoGradGuard no_grad;  // this is so PyTorch doesn't get mad when we reassign tensors
    int i;
    torch::Tensor in, hidden, prediction;
    TorchModels::TorchRNN model(histLength, numLayers, dtype);
    int batchSize = 10;
    hidden = model.makeHiddenState(batchSize);
    std::cout << "Check test prediction succeeds" << std::endl;
    std::cout << "Starting prediction loop\n";
    for (i = 0 ; i < numLoops ; ++i) {
        in = torch::randn({batchSize, 1, histLength});
        std::cout << "Made random input tensor\n";
        try {
            std::tie(prediction, hidden) = model.forward(in, hidden);
        } catch (std::runtime_error e) {
            std::cout << "Error occurred in inference." << e.what() << std::endl;
            exit(-1);
        }
        std::cout << "recieved output of model forward\n";
        std::cout << prediction << std::endl;
    }

    return 0;
}

int testRandomHiddenSucceeds(int histLength, int numLayers, torch::Dtype dtype) {
    TorchModels::TorchRNN model(histLength, numLayers, dtype);
    int batched;
    bool bidirect;
    torch::Tensor hidden, testTensor;

    batched = 0;
    bidirect = false;
    testTensor = torch::ones({numLayers, model.getHiddenFeatures()});
    hidden = model.makeHiddenState(batched, bidirect);
    if ( hidden.is_same_size(testTensor) )
        return 0;
    return 1;
}