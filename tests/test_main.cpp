//
// Created by zhymi on 9/18/2023.
//

#include "test_main.h"
int main(int argc, char* argv[]) {
    cout << "Starting tests" << endl;
    torchRNNTests();
    forecasterTests();
    weightHolderTests();
    queueTests();
    cout << "End of tests" << endl;
    return 0;
}

void torchRNNTests() {
    cout << "Beginning TorchRNNTests" << endl;
    int testModelConstruct1, testModelConstruct2, testForward1, testRandomHidden1;
    int histLength, numLayers, numLoops;
    torch::Dtype dtype;
    histLength = 7;
    numLayers = 3;
    numLoops = 10;
    dtype = torch::kFloat32;
    cout << "Starting Model Construction tests\n";
    // test model construction
    testModelConstruct1 = testModelEmptyConstructionSucceeds();
    testModelConstruct2 = testModelParameterConstructionSucceeds(histLength, numLayers, dtype);
    if ( testModelConstruct1 + testModelConstruct2 > 0 )
        cout << "Model construction tests failed: (" << testModelConstruct1 << ", " << testModelConstruct2 << ")\n";
    cout << "Starting model method tests" << endl;
    // test methods
    testRandomHidden1 = testRandomHiddenSucceeds(histLength, numLayers, dtype);
    if ( testRandomHidden1 != 0 )
        cout << "RandomHidden failed with error code: " << testRandomHidden1 << endl;
    cout << "Test 2\n";
    testForward1 = testForwardSucceeds(histLength, numLayers, dtype, numLoops);
    if ( testForward1 != 0 )
        cout << "Forward call failed with error code: " << testForward1 << endl;
    cout << "End of torchRNNTests" << endl;
}

void forecasterTests() {
    cout << "Beginning forecaster tests" << endl;
    std::shared_ptr<TorchModels::TorchRNN> model1(new TorchModels::TorchRNN());
    std::shared_ptr<TorchModels::TorchRNN> model2(new TorchModels::TorchRNN());
    int testUpdate2 = testUpdateWeightsRNNToRNN(model1, model2);
    cout << "Test on model shared pointers: " << testUpdate2 << endl;
    int numSamples, batchSize, histLength, testPredict1, testPredict2;
    numSamples = 10;
    batchSize = 1;
    histLength = 1;
    testPredict1 = testTorchRNNMakePredictionsTorchPipeSucceeds(histLength, numSamples, batchSize);
    cout << "First model prediction: " << testPredict1 << endl;
    testPredict2 = testTorchRNNMakeImprovementsTorchPipeSucceeds(histLength, numSamples, batchSize);
    cout << "Model improvement results: " << testPredict2 << endl;
}

void weightHolderTests() {
    int test1, test2, test3;
    test1 = testReadHolderSucceeds();
    test2 = testReadHolderIsNewFalse();
//    test3 = testWriteHolder();
    cout << "Test 1 result: " << test1 << endl;
    cout << "Test 2 result: " << test2 << endl;
}

void queueTests() {
    cout << "Start of queue tests" << endl;
    int test1, test2;
    test1 = testPushIntegers();
}