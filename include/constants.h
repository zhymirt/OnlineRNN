//
// Created by zhymi on 6/13/2023.
//

#ifndef ONLINERNN_CONSTANTS_H
#define ONLINERNN_CONSTANTS_H

#include <limits>
#pragma warning(push, 0)
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#pragma warning(pop)

namespace constants
{
    static const double FloatInf = std::numeric_limits<double>::infinity();
// Two means "stop"
    const torch::Tensor queueSentinel = torch::full((2), -FloatInf);
    const torch::Tensor pipeSentinel = torch::full((2), -FloatInf);
// One is "next"
    const torch::Tensor pipeConfirmation = torch::full((1), -FloatInf);

    const std::string noParams = "";
    const std::string paramSentinel = "NO_MORE_PARAMS";

    const torch::OrderedDict<std::string, torch::Tensor> noDict; // Todo define this
    const torch::OrderedDict<std::string, torch::Tensor> OrdDictSentinel;
}
#endif //ONLINERNN_CONSTANTS_H
