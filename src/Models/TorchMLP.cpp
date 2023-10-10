//
// Created by zhymi on 9/25/2023.
//

#include "Models/TorchMLP.h"

namespace TorchModels {
    TorchMLP::TorchMLP() {
        this->setInputLength(1);
        this->setDtype(torch::kFloat32);
        int hidden1, hidden2, hidden3, hidden4;
        hidden1 = 8;
        hidden2 = 8;
        hidden3 = 16;
        hidden4 = 16;
        this->firstLayer = register_module(
                "firstLayer", torch::nn::Linear(torch::nn::LinearOptions(1, hidden1))
        );
        this->layer2 = register_module(
                "layer2", torch::nn::Linear(torch::nn::LinearOptions(hidden1, hidden2))
        );
        this->layer3 = register_module(
                "layer3", torch::nn::Linear(torch::nn::LinearOptions(hidden2, hidden3))
        );
        this->layer4 = register_module(
                "layer4", torch::nn::Linear(torch::nn::LinearOptions(hidden3, hidden4))
        );
        this->lastLayer = register_module(
                "lastLayer", torch::nn::Linear(torch::nn::LinearOptions(hidden4, 1))
        );
    }
    TorchMLP::TorchMLP(int inputSize, torch::Dtype dataType) {
        this->setInputLength(inputSize);
        this->setDtype(dataType);
        int hidden1, hidden2, hidden3, hidden4;
        hidden1 = 8;
        hidden2 = 8;
        hidden3 = 16;
        hidden4 = 16;
        this->firstLayer = register_module(
                "firstLayer", torch::nn::Linear(torch::nn::LinearOptions(inputSize, hidden1))
                );
        this->layer2 = register_module(
                "layer2", torch::nn::Linear(torch::nn::LinearOptions(hidden1, hidden2))
                );
        this->layer3 = register_module(
                "layer3", torch::nn::Linear(torch::nn::LinearOptions(hidden2, hidden3))
                );
        this->layer4 = register_module(
                "layer4", torch::nn::Linear(torch::nn::LinearOptions(hidden3, hidden4))
                );
        this->lastLayer = register_module(
                "lastLayer", torch::nn::Linear(torch::nn::LinearOptions(hidden4, 1))
                );
    }
    std::tuple<torch::Tensor, torch::Tensor> TorchMLP::forward(torch::Tensor inputVal) {
        torch::Tensor out, standin;
        out = torch::relu(this->firstLayer->forward(inputVal));
        out = torch::relu(this->layer2->forward(out));
        out = torch::relu(this->layer3->forward(out));
        out = torch::relu(this->layer4->forward(out));
        out = torch::tanh(this->lastLayer->forward(out));
        return std::tie(out, standin);
    }
    inferenceTuple TorchMLP::forward(torch::Tensor inputVal, torch::Tensor hidden) {
        return this->forward(inputVal);
    }
    std::tuple<torch::Tensor, torch::Tensor> TorchMLP::predict(torch::Tensor inputVal, torch::Tensor hidden) {
        return this->forward(inputVal);
    }

    int TorchMLP::getInputLength() const {
        return this->inputLength;
    }

    void TorchMLP::setInputLength(int inputLength) {
        this->inputLength = inputLength;
    }

    torch::Dtype TorchMLP::getDtype() const {
        return this->dtype;
    }

    void TorchMLP::setDtype(torch::Dtype dtype) {
        this->dtype = dtype;
    }

    torch::Tensor TorchMLP::loss(torch::Tensor prediction, torch::Tensor actual) {
        return torch::empty_like(actual); // not implemented todo make this constant or remove overall
    }

    torch::Tensor TorchMLP::randomHidden(int batched, bool bidirect) {
        return torch::empty(0);  // todo make this more broken
    }

    torch::Tensor TorchMLP::makeHiddenState(int batched, bool bidirect) {
        return this->randomHidden(batched, bidirect);
    }
} // TorchModels