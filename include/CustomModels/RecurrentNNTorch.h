//
// Created by zhymi on 5/24/2023.
//

#ifndef ONLINERNN_RECURRENTNNTORCH_H
#define ONLINERNN_RECURRENTNNTORCH_H
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)

class RecurrentNeuralNetworkTorch : public torch::nn::Module {
public:
    virtual ~RecurrentNeuralNetworkTorch() {}
//    virtual std::vector<torch::Tensor> parameters(bool recurse=true) = 0;
//    virtual torch::OrderedDict<std::string, torch::Tensor> named_parameters(bool recurse = true) = 0;
//    virtual torch::OrderedDict<std::string, torch::Tensor> named_buffers(bool recurse = true) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor inputVal, torch::Tensor hiddenState) = 0;
    virtual torch::Tensor loss(torch::Tensor prediction, torch::Tensor actual) = 0;
    virtual torch::Tensor makeHiddenState() = 0;
};
#endif //ONLINERNN_RECURRENTNNTORCH_H
