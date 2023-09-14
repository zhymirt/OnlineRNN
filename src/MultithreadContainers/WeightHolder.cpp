//
// Created by zhymi on 8/29/2023.
//

#include "WeightHolder.h"

WeightHolder::WeightHolder() {
    this->isNew = false;
    this->params = constants::noDict;
    this->buffers = constants::noDict;
}

regType WeightHolder::getParams() {
    return this->params;
}
regType WeightHolder::getBuffers() {
    return this->buffers;
}

void WeightHolder::setParams(regType otherParams) {
    this->params = otherParams;
}
void WeightHolder::setBuffers(regType otherBuffers) {
    this->buffers = otherBuffers;
}

std::tuple<regType, regType> WeightHolder::readHolder() {
    regType newBuffers, newParams;
    std::lock_guard<std::mutex> lock(this->mtx);
    if ( this->isNew ) {
        newBuffers = this->getBuffers();
        newParams = this->getParams();
        this->isNew = false;
    } else{
        newBuffers = newParams = constants::noDict;
    }
    return std::tuple<regType, regType>(newBuffers, newParams);
}
void WeightHolder::writeHolder(regType otherParams, regType otherBuffers) {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->setParams(otherParams);
    this->setBuffers(otherBuffers);
    this->isNew = true;
}
bool WeightHolder::newMessage() {
    std::lock_guard<std::mutex> lock(this->mtx);
    return this->isNew;
}
bool WeightHolder::endUpdates() {
    std::lock_guard<std::mutex> lock(this->mtx);
    return this->isDone;
}