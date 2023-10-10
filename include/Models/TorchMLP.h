//
// Created by zhymi on 9/25/2023.
//

#ifndef ONLINERNN_TORCHMLP_H
#define ONLINERNN_TORCHMLP_H

#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)
#include "CustomModels/RecurrentNNTorch.h"

namespace TorchModels {

    class TorchMLP : public RecurrentNeuralNetworkTorch {
    private:
        torch::Dtype dtype;
        int inputLength;
        torch::nn::Linear firstLayer{nullptr}, lastLayer{nullptr}, layer2{nullptr},
        layer3{nullptr}, layer4{nullptr};
    public:
        TorchMLP();
        TorchMLP(int histLength, torch::Dtype dataType=torch::kFloat32);
        torch::Dtype getDtype() const;
        void setDtype(torch::Dtype dtype);
        int getInputLength() const;
        void setInputLength(int inputLength);

        inferenceTuple forward(torch::Tensor inputVal);
        inferenceTuple forward(torch::Tensor inputVal, torch::Tensor hidden);
        inferenceTuple predict(torch::Tensor inputVal, torch::Tensor hidden);
        torch::Tensor loss(torch::Tensor prediction, torch::Tensor actual);
        torch::Tensor randomHidden(int batched, bool bidirect=false);
        torch::Tensor makeHiddenState(int batched=0, bool bidirect=false);
    };

} // TorchModels

#endif //ONLINERNN_TORCHMLP_H
