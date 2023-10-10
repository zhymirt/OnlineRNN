//
// Created by zhymi on 6/22/2023.
//

#ifndef ONLINERNN_WEIGHTHOLDER_H
#define ONLINERNN_WEIGHTHOLDER_H

#include <string>
#include <mutex>
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)
#include "constants.h"

typedef torch::OrderedDict<std::string, torch::Tensor> regType;
class WeightHolder {
private:
    std::mutex mtx;
    regType params, buffers;
    bool isNew, isDone;

public:
    bool getIsNew();
    bool getIsDone();
    regType getParams();
    regType getBuffers();
    void setIsNew(bool setIsNew);
    void setIsDone(bool setIsDone);
    void setParams(regType otherParams);
    void setBuffers(regType otherBuffers);
    WeightHolder();
    std::tuple<regType, regType> readHolder();
    void writeHolder(regType newParams, regType newBuffers);
    bool newMessage();
    bool endUpdates();
};
#endif //ONLINERNN_WEIGHTHOLDER_H
