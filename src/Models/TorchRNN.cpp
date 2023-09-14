//
// Created by zhymi on 8/30/2023.
//

#include "TorchRNN.h"

namespace TorchModels {
    TorchRNN::TorchRNN() {
        // magic numbers for default constructor
        this->setInputLength(1);
        this->setNumLayers(1);
        this->setHiddenFeatures(64); // magic number; should remove
        this->setDtype(torch::kFloat32);
        // end magic numbers for default constructor
        this->rnn = register_module(
                "rnn", torch::nn::RNN(
                        torch::nn::RNNOptions(
                                this->getInputLength(), this->getHiddenFeatures())
                                .num_layers(this->getNumLayers()).batch_first(true)
                                .bias(false).nonlinearity(torch::kTanh)));
        this->lastLayer = register_module(
                "fcc1", torch::nn::Linear(
                        torch::nn::LinearOptions(
                                this->getHiddenFeatures(), 1).bias(false)));

    }

    TorchRNN::TorchRNN(int histLength, int numLayers, torch::Dtype dataType) {
        this->setInputLength(histLength);
        this->setNumLayers(numLayers);
        this->setHiddenFeatures(64); // magic number; should remove
        this->setDtype(dataType);
        this->rnn = register_module(
                "rnn", torch::nn::RNN(
                        torch::nn::RNNOptions(
                                this->getInputLength(), this->getHiddenFeatures())
                                .num_layers(this->getNumLayers()).batch_first(true)
                                .bias(false).nonlinearity(torch::kTanh)));
        this->lastLayer = register_module(
                "fcc1", torch::nn::Linear(
                        torch::nn::LinearOptions(
                                this->getHiddenFeatures(), 1).bias(false)));

    }

    std::tuple<torch::Tensor, torch::Tensor> TorchRNN::forward(torch::Tensor inputVal, torch::Tensor hidden) {
        std::tuple<torch::Tensor, torch::Tensor> out = this->rnn->forward(inputVal, hidden);
        return out;
    }

    std::tuple<torch::Tensor, torch::Tensor> TorchRNN::predict(torch::Tensor inputVal, torch::Tensor hidden) {
        return this->forward(inputVal, hidden);
    }

    int TorchRNN::getInputLength() const {
        return this->inputLength;
    }

    void TorchRNN::setInputLength(int inputLength) {
        this->inputLength = inputLength;
    }

    int TorchRNN::getHiddenFeatures() const {
        return this->hiddenFeatures;
    }

    void TorchRNN::setHiddenFeatures(int hiddenFeatures) {
        this->hiddenFeatures = hiddenFeatures;
    }

    int TorchRNN::getNumLayers() const {
        return this->numLayers;
    }

    void TorchRNN::setNumLayers(int numLayers) {
        this->numLayers = numLayers;
    }

    torch::Dtype TorchRNN::getDtype() const {
        return this->dtype;
    }

    void TorchRNN::setDtype(torch::Dtype dtype) {
        this->dtype = dtype;
    }

    torch::Tensor TorchRNN::loss(torch::Tensor prediction, torch::Tensor actual) {
        return torch::empty_like(actual); // not implemented todo make this constant or remove overall
    }

    torch::Tensor TorchRNN::randomHidden(int batched, bool bidirect) {
        int vecSize = !bidirect ? this->getNumLayers() : 2 * this->getNumLayers();
        if (batched != 0)
            return torch::randn({vecSize, batched, this->getHiddenFeatures()});
        return torch::randn({vecSize, this->getHiddenFeatures()});
    }

    torch::Tensor TorchRNN::makeHiddenState(int batched, bool bidirect) {
        return this->randomHidden(batched, bidirect);
    }

} // TorchModels