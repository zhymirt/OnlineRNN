#include "forecaster.h"

void make_predictions_torch_pipe(
        torch::jit::script::Module model, QueueType inputQueue, QueueType outputQueue,
        int *pipe) {
	int count;
	torch::Tensor data, hidden;
	data = inputQueue.get(); // change this later, it won't work
	hidden = model.make_hidden_state(); // TODO Make these classes
	while (data != null) {
		++count;
		auto [prediction, hidden] = model.predict(data, hidden);
		
	}
}

void makePredictionsTorchAllPipes(
        ModuleType model, PipeType
        ) {
    at::Tensor data;
    try {
        data = dataPipe
    } catch () {

    }
}

void torch_update_state(torch::jit::script::Module model, std::stringstream new_state) {
	torch::load(model, new_state);
}

//void push_data_to_queues(torch::Tensor data, int start, int end_index,
//		int time_skip, int input_size, ) {
//
//}
// TODO this function will return a pair of vectors
// TODO make two-way pipe class
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
