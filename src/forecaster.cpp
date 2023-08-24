#include "forecaster.h"


std::tuple<at::Tensor, at::Tensor> runForecaster(
        ModuleType learner, ModuleType predictor, torch::Tensor data,
        int inputSize, int timeSkip) { // Assume this one will use torch, another might not
    int seqLength, start, end, endIndex;
    // instantiate variables
    // get end index
    endIndex = data.size(1) - end;
    LossFunction loss_fn = torch::nn::MSELoss();
    auto optimizer = std::make_shared<torch::optim::Adam>(learner->parameters());// (learner.parameters());
    QueueType predictorInQueue, predictorOutQueue, learnerOutQueue;
    LearnQueueType learnerInQueue;
    PipeType paramPipe = std::make_shared<StringHolder>(); // create param pipe as shared_ptr
    predictorInQueue = std::make_shared<MultithreadQueue<torch::Tensor>>();
    predictorOutQueue = std::make_shared<MultithreadQueue<torch::Tensor>>();
    learnerInQueue = std::make_shared<MultithreadQueue<std::tuple<torch::Tensor, torch::Tensor>>>();
    learnerOutQueue = std::make_shared<MultithreadQueue<torch::Tensor>>();
    std::thread pThread(
            make_predictions_torch_pipe, predictor, predictorInQueue,
            predictorOutQueue, paramPipe);
    std::thread lThread(
            make_improvements_torch_pipe, learner, learnerInQueue, loss_fn,
            seqLength, optimizer, paramPipe, learnerOutQueue);
    push_data_to_queues(
            data, start, endIndex, timeSkip, inputSize,
            predictorInQueue, learnerInQueue);
    pThread.join();
    lThread.join();

}
void make_predictions_torch_pipe(
        ModuleType model, QueueType inputQueue, QueueType outputQueue,
        StringHolder paramPipe) {
    torch::NoGradGuard no_grad;
    int maxSize = SSIZE_MAX;
    int count = 0;
    std::string newParams;
    torch::Tensor data, prediction, hidden;
    // TODO make try catch of all reads and writes
    data = inputQueue->front();
    inputQueue->pop();
    hidden = model->makeHiddenState(); // TODO Make these classes
    while (!data.equal(constants::queueSentinel)) {  // while data is not sentinel
        ++count;
        // start torch no grad
		auto [newPrediction, newHidden] = model->predict(data, hidden);
//        hidden = newHidden;
        prediction = newPrediction;
        hidden = newHidden;
        // delete newPrediction and newHidden?
        outputQueue->push(prediction);
        if ( paramPipe.newMessage() ) { // todo change this to check for pipe poll
            newParams = paramPipe.readString();
            update_weights(model, newParams);  // todo need new pipe impl for weights and buffers
            // delete newParams?
            // No need to send pipe confirmation since sender doesn't need to wait
        }
        data = inputQueue->front();
        inputQueue->pop();
    }
    // delete data
    outputQueue->push(constants::queueSentinel);
    // finally
    // delete paramPipe? not shared so idk
    // would need to close queue, but probably just delete?
}/*
void make_predictions_torch_pipe(
        ModuleType model, QueueType inputQueue, QueueType outputQueue,
        int *pipe) {
    int maxSize = SSIZE_MAX;
	int count = 0;
	torch::Tensor data, hidden;
    // TODO make try catch of all reads and writes
	int readCode = read(inputQueue, &data, maxSize); // change this later, it won't work
	hidden = model.makeHiddenState(); // TODO Make these classes
	while (data != null) {
		++count;
//		auto [prediction, newHidden] = model.predict(data, hidden);
        hidden = newHidden;
        write(outputQueue, prediction);



		
	}
}*/



void make_improvements_torch_pipe(
        ModuleType model, LearnQueueType in_queue, LossFunction loss_fn, int seq_length,
        OptimizerType optimizer, PipeType paramPipe, QueueType out_queue) {
    bool predictorDone, updatePredictor, newWeights;
    torch::Tensor data, actual, hidden;
    int count = 0;
    predictorDone = false;
    updatePredictor = newWeights = true;
    auto [newData, newActual] = in_queue->front();
    in_queue->pop();
    data = newData;
    actual = newActual;
    hidden = model->makeHiddenState();
    while ( !data.equal(constants::queueSentinel) && !predictorDone ) {
        auto [prediction, newHidden] = model->predict(data, hidden);
        out_queue->push(prediction);
        if (count % seq_length == 0) {
            torch::Tensor loss = loss_fn(prediction, actual);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            hidden = model->makeHiddenState();
            newWeights = true;
        } else {
            hidden = newHidden;
        }
        // delete data, and actual?
        // doing pipe stuff
        // end pipe stuff
    }
    out_queue->push(constants::queueSentinel);
    // finally
    // close pipe and queue?
}

//void make_improvements_torch_pipe(
//        ModuleType model, LearnQueueType in_queue, LossFunction loss_fn, int seq_length,
//        OptimizerType optimizer, PipeType paramPipe);

void update_weights(
        ModuleType model, torch::OrderedDict<std::string, torch::Tensor> params,
        torch::OrderedDict<std::string, torch::Tensor> buffers) {
    // no way this is allowed
//    model->named_parameters() = params;
//    model->named_buffers() = buffers;
// actual attempt here
    assert(model->named_parameters() == params.size());  // assert same number of key/value pairs
    assert(model->named_buffers() == buffers.size());  // assert same number of key/value pairs
    // iterate and modify all named_parameters
    for (std::string key : model->named_parameters().keys()) {
        assert(params.find(key) != null_ptr);
        model->named_parameters()[key] = *params.find(key);
    }
    // iterate and modify all named_buffers
    for (std::string key : model->named_buffers().keys()) {
        assert(buffers.find(key) != null_ptr);
        model->named_buffers()[key] = *buffers.find(key);
    }
}
//void torch_update_state(ModuleType model, std::stringstream new_state) {
//    torch::load(torch::nn::ModuleHolder(model), new_state);
//	torch::load(new_state, model);
//}

void push_data_to_queues(const torch::Tensor &data, int start, int endIndex,
		int time_skip, int input_size, QueueType currQueue, LearnQueueType histQueue) {
    int idx, end;
    int currOffset, endOffset;
    endOffset = time_skip + (input_size * 2);
    currOffset = input_size + time_skip;
    for (idx = start ; idx < endIndex ; ++idx) {
        end = idx + endOffset;
        currQueue->push(data.index({Ellipsis, Slice((idx + currOffset), end)}).detach());
        histQueue->push(
                std::make_tuple(
                        data.index({Ellipsis, Slice(idx, idx + input_size)}).detach(),
                        data.index({Ellipsis, end}).unsqueeze(1).detach()));
        // sleep for 0.05s
    }
    currQueue->push(constants::queueSentinel);
    histQueue->push(std::make_tuple(constants::queueSentinel, constants::queueSentinel));
}
// TODO this function will return a pair of vectors
// TODO make two-way pipe class
/*
int* push_data_to_pipes(torch::Tensor data, int start,
		int endIndex, int timeSkip, int inputSize, int currReadPipe,
	int currWritePipe, int histReadPipe, int histWritePipe) {
	bool currBuffEmpty, histBuffEmpty, currReady, histReady;
	pipeBuffer currBuff, histBuff, histXBuff, histYBuff;
	pipeBuffer currOut, histOut;
	try {
		for (int idx = start ; idx < endIndex ; ++idx) {
			int end = idx + timeSkip + (inputSize * 2);
			// append to curr_buff
			// append to hist_x_buff
			// append to hist_y_buff
			if (currReady == true) {
				// send through curr_pipe
				write(currWritePipe, &currBuff, );
				currBuff.clear();
				currReady = false;
			}
			if (histReady == true) {
				// send through hist_pipe
				histXBuff.clear();
				histYBuff.clear();
				histReady = false;
			}
		}
		if (currBuff.empty()) {

		}
	} catch (...) {
        std::cout << "Something went wrong" << std::endl;
    }
    // TODO fix this
	closePipe(currWritePipe[1]);
    closePipe(currReadPipe[0]);
}
 */
