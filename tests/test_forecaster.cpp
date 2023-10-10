//
// Created by zhymi on 8/29/2023.
//
#include "test_forecaster.h"

/*
int main(int argc, char* argv[]) {
    cout << "Beginning test" << endl;
    std::shared_ptr<TorchModels::TorchRNN> model1(new TorchModels::TorchRNN());
    std::shared_ptr<TorchModels::TorchRNN> model2(new TorchModels::TorchRNN());
    int testUpdate2 = testUpdateWeightsRNNToRNN(model1, model2);
    cout << "Test on model shared pointers: " << testUpdate2 << endl;
    return 0;
} */

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
//int testTorchMLPMakePredictionsTorchPipeSucceeds(int histLength, int numSamples, int batchSize) {
//    QueueType inQueue, outQueue;
//    WeightShareType weightHolder;
//    torch::Tensor hidden, prediction, in;
//
//    sharedModule model(new TorchModels::TorchMLP(histLength));
//    for (int i = 0 ; i < numSamples ; ++i) { // insert elements into queue
//        in = torch::randn({batchSize, 1,  histLength});
//        inQueue->push(in);
//    }
//    make_predictions_torch_pipe(model, inQueue, outQueue, weightHolder);
//    return inQueue->empty() ? 0 : 1;
//}
int testTorchRNNMakePredictionsTorchPipeSucceeds(int histLength, int numSamples, int batchSize) {
    QueueType inQueue(new MultithreadQueue<torch::Tensor>);
    QueueType outQueue(new MultithreadQueue<torch::Tensor>);
    WeightShareType weightHolder(new WeightHolder());
    torch::Tensor hidden, prediction;

    sharedModule model(new TorchModels::TorchRNN(histLength));
    cout << "pushing items to queue" << endl;
    for (int i = 0 ; i < numSamples ; ++i) { // insert elements into queue
        torch::Tensor in = torch::randn({batchSize, 1,  histLength});
        inQueue->push(in.detach());
    }
    inQueue->push(constants::queueSentinel);
    cout << "items pushed to queue" << endl;
//    exit(0);
    try {
        make_predictions_torch_pipe(model, inQueue, outQueue, weightHolder);
    } catch (std::runtime_error e) {
        cout << "runtime error occurred: " << e.what() << endl;
        exit(-10);
    }
    return inQueue->empty() ? 0 : 1;
}
int testMakePredictionsTorchPipeSucceeds(sharedModule model, QueueType inQueue, QueueType outQueue, WeightShareType weightHolder) {
    make_predictions_torch_pipe(model, inQueue, outQueue, weightHolder);
    return inQueue->empty() ? 0 : 1;
}

int testTorchRNNMakeImprovementsTorchPipeSucceeds(int histLength, int numSamples, int batchSize) {
    LearnQueueType inQueue(new MultithreadQueue<std::tuple<torch::Tensor, torch::Tensor>>);
    QueueType outQueue(new MultithreadQueue<torch::Tensor>);
    WeightShareType weightHolder(new WeightHolder());
    torch::Tensor hidden, prediction;
    int seqLength = std::min(numSamples, 4);
    torch::nn::MSELoss lossFn;

    sharedModule model(new TorchModels::TorchRNN(histLength));
    OptimizerType optimizer(new torch::optim::Adam(model->parameters()));
    cout << "pushing items to queue" << endl;
    for (int i = 0 ; i < numSamples ; ++i) { // insert elements into queue
        torch::Tensor in = torch::randn({batchSize, 1,  histLength});
        torch::Tensor in2 = torch::randn_like(in);
        auto dataTuple = std::tuple(in.detach(), in2.detach());
        inQueue->push(dataTuple);
    }
    inQueue->push(std::tuple(constants::queueSentinel, constants::queueSentinel));
    cout << "items pushed to queue" << endl;
//    exit(0);
    try {
        make_improvements_torch_pipe(model, inQueue, lossFn, seqLength, optimizer, weightHolder, outQueue);
    } catch (std::runtime_error e) {
        cout << "runtime error occurred: " << e.what() << endl;
        exit(-10);
    }
    return inQueue->empty() ? 0 : 1;
}