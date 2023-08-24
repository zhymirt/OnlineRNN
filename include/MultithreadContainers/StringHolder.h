//
// Created by zhymi on 6/13/2023.
//

#ifndef ONLINERNN_STRINGHOLDER_H
#define ONLINERNN_STRINGHOLDER_H

#include <string>
#include <mutex>
#include "constants.h"

class StringHolder {
private:
    std::mutex strMtx;
    std::string params;
    bool isNew;
    std::string getString();
    void setString(std::string aString);

public:
    StringHolder();
    std::string readString();
    void writeString(std::string newParams);
    bool newMessage();
};
#endif //ONLINERNN_STRINGHOLDER_H
