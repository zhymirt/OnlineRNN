//
// Created by zhymi on 9/18/2023.
//

#ifndef ONLINERNN_TEST_MAIN_H
#define ONLINERNN_TEST_MAIN_H

#include <iostream> // cout, cin, endl, etc.
#include "Models/testTorchRNN.h"
#include "test_forecaster.h"
#include "Models/TorchRNN.h"
#include "test_weightHolder.h"
#include "test_multithreadQueue.h"

using std::cout;
using std::endl;

void torchRNNTests();
void forecasterTests();
void weightHolderTests();
void queueTests();


#endif //ONLINERNN_TEST_MAIN_H
