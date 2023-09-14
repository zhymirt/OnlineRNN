//
// Created by zhymi on 6/13/2023.
//

#ifndef ONLINERNN_STRINGHOLDER_CPP
#define ONLINERNN_STRINGHOLDER_CPP

#include "StringHolder.h"


StringHolder::StringHolder() {
    this->isNew = false;
    this->params = constants::noParams;
}
std::string StringHolder::getString() { // get string, doesn't use lock
    return this->params;
}
void StringHolder::setString(std::string aString) {
    this->params = aString;
}
std::string StringHolder::readString() {  // lock mutex, read string, mark as read
    std::string rString;
    std::lock_guard<std::mutex> lock(this->strMtx);
    if ( this->isNew ) {  // if new model, copy new params and mark as not new
        rString = this->getString();
        this->isNew = false;
    } else {
        rString = constants::noParams;  // sets to empty string so model knows not to copy, may change later
    }
    return rString;
}
void StringHolder::writeString(std::string newParams) {
    std::lock_guard<std::mutex> lock(this->strMtx);
    this->setString(newParams);
    this->isNew = true;
}
bool StringHolder::newMessage() {
    std::lock_guard<std::mutex> lock(this->strMtx);
    return this->isNew;
}

#endif //ONLINERNN_STRINGHOLDER_CPP
