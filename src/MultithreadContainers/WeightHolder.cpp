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
bool WeightHolder::getIsNew() {
    return this->isNew;
}

bool WeightHolder::getIsDone() {
    return this->isDone;
}

void WeightHolder::setParams(regType otherParams) {
    this->params = otherParams;
}
void WeightHolder::setBuffers(regType otherBuffers) {
    this->buffers = otherBuffers;
}

void WeightHolder::setIsDone(bool setIsDone) {
    this->isDone = setIsDone;
}

void WeightHolder::setIsNew(bool setIsNew) {
    this->isNew = setIsNew;
}

std::tuple<regType, regType> WeightHolder::readHolder() {
    regType newParams, newBuffers;
    std::lock_guard<std::mutex> lock(this->mtx);
    if ( this->isNew ) {
        newParams = this->getParams();
        newBuffers = this->getBuffers();
        this->isNew = false;
    } else {
        newParams = newBuffers = constants::noDict;
    }
    // order of tuple is parameters then buffers
    return std::tuple<regType, regType>(newParams, newBuffers);
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