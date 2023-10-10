//
// Created by zhymi on 10/3/2023.
//

#include "test_multithreadQueue.h"

int testPushIntegers() {
    int end = 10;
    MultithreadQueue<int> myQueue;
    for (int i = 0 ; i < end ; ++i)
        myQueue.push(i);
    return 0;
}

int testPushTensors() {
    return -1;
}
