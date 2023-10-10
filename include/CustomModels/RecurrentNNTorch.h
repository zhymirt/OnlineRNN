//
// Created by zhymi on 5/24/2023.
//

#ifndef ONLINERNN_RECURRENTNNTORCH_H
#define ONLINERNN_RECURRENTNNTORCH_H
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)

typedef std::tuple<torch::Tensor, torch::Tensor> inferenceTuple;

class RecurrentNeuralNetworkTorch : public torch::nn::Module {
public:
    virtual ~RecurrentNeuralNetworkTorch() {}
    virtual inferenceTuple predict(torch::Tensor inputVal, torch::Tensor hiddenState) = 0;
    virtual torch::Tensor loss(torch::Tensor prediction, torch::Tensor actual) = 0;
    virtual torch::Tensor makeHiddenState(int batched=0, bool bidirect=false) = 0;
};
#endif //ONLINERNN_RECURRENTNNTORCH_H
