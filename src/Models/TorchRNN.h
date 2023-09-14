//
// Created by zhymi on 8/30/2023.
//

#ifndef ONLINERNN_TORCHRNN_H
#define ONLINERNN_TORCHRNN_H

#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)

#include "RecurrentNNTorch.h"
//#include "CustomModels/RecurrentNNTorch.h"

namespace TorchModels {

    class TorchRNN : public RecurrentNeuralNetworkTorch {
    private:
        torch::Dtype dtype;
    public:

    private:
        int inputLength, hiddenFeatures, numLayers;
        torch::Tensor hiddenState;
        torch::nn::RNN rnn{nullptr};
        torch::nn::Linear lastLayer{nullptr};
    public:
        TorchRNN();
        TorchRNN(int histLength, int numLayers=1, torch::Dtype dataType=torch::kFloat32);
        torch::Dtype getDtype() const;
        void setDtype(torch::Dtype dtype);
        int getInputLength() const;

        void setInputLength(int inputLength);

        int getHiddenFeatures() const;

        void setHiddenFeatures(int hiddenFeatures);

        int getNumLayers() const;

        void setNumLayers(int numLayers);

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inputVal, torch::Tensor hidden);
        std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor inputVal, torch::Tensor hidden);
        torch::Tensor loss(torch::Tensor prediction, torch::Tensor actual);
        torch::Tensor randomHidden(int batched, bool bidirect=false);
        torch::Tensor makeHiddenState(int batched=0, bool bidirect=false);
//        ~TorchRNN();
    };

} // TorchModels

#endif //ONLINERNN_TORCHRNN_H
