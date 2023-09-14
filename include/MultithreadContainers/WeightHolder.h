//
// Created by zhymi on 6/22/2023.
//

#ifndef ONLINERNN_WEIGHTHOLDER_H
#define ONLINERNN_WEIGHTHOLDER_H

#include <string>
#include <mutex>
#include "constants.h"

typedef torch::OrderedDict<std::string, torch::Tensor> regType;
class WeightHolder {
private:
    std::mutex mtx;
    regType params, buffers;
    bool isNew, isDone;
    regType getParams();
    regType getBuffers();
    void setParams(regType otherParams);
    void setBuffers(regType otherBuffers);
public:
    WeightHolder();
    std::tuple<regType, regType> readHolder();
    void writeHolder(regType newParams, regType newBuffers);
    bool newMessage();
    bool endUpdates();
};
#endif //ONLINERNN_WEIGHTHOLDER_H
