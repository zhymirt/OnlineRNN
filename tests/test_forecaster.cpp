//
// Created by zhymi on 8/29/2023.
//
#include "test_forecaster.h"

int main(int argc, char* argv[]) {
    cout << "Beginning test" << endl;
    std::shared_ptr<TorchModels::TorchRNN> model1(new TorchModels::TorchRNN());
    std::shared_ptr<TorchModels::TorchRNN> model2(new TorchModels::TorchRNN());
    int testUpdate2 = testUpdateWeightsRNNToRNN(model1, model2);
    cout << "Test on model shared pointers: " << testUpdate2 << endl;
    return 0;
}

int testUpdateWeightsRNNToRNN(sharedModule model1, sharedModule model2) { // test on shared ptrs
    update_weights(
            model2, model1->named_parameters(),
            model1->named_buffers());
    const auto& otherParams = model2->named_parameters();
    for (const auto& pair : model1->named_parameters()) {
        // if key in model2 not in model1 or the values are not equal, return failure
        if ( !otherParams.contains(pair.key()) )
            return 1;
        if ( !pair.value().equal(otherParams[pair.key()]) ) {
            cout << pair.value() << " doesn't equal " << otherParams[pair.key()] << endl;
            return 2;
        }
    }
    return 0;
}