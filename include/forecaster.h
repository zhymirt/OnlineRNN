#ifndef FORECASTER_H
#define FORECASTER_H

#include <iostream> // cout, cin, endl, etc.
// torch
//#include <torch/script.h> // One-stop header.
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)

#include "MultithreadQueue.h" // multithreading queue
#include "constants.h"  // constant values used in code
#include "RecurrentNNTorch.h"  // Base class for recurrent models used in functions
#include "StringHolder.h"  // multithreading string holder for model parameters

#if _MSC_VER
#include <Windows.h>
const auto closePipe = CloseHandle;
#else
#include <unistd.h>
const auto closePipe = close;
#endif

using std::vector;
using namespace torch::indexing;
//using torch::optim::Adam;
//using torch::optim::SGD;
//using torch::nn::MSELoss;

//typedef vector<at::Tensor> pipeBuffer;
typedef std::shared_ptr<MultithreadQueue<torch::Tensor>> QueueType;
typedef std::shared_ptr<MultithreadQueue<std::tuple<torch::Tensor, torch::Tensor>>> LearnQueueType;
typedef std::shared_ptr<RecurrentNeuralNetworkTorch> ModuleType;
typedef std::shared_ptr<StringHolder> PipeType;
//typedef torch::jit::script::Module ModuleType;
//typedef int PipeType;


//typedef at::Tensor (*loss_function)(at::Tensor, at::Tensor);
typedef torch::nn::ModuleHolder<torch::nn::MSELossImpl> LossFunction;  // MSE
typedef std::shared_ptr<torch::optim::Optimizer> OptimizerType;  // Adam or SGD


void make_predictions_torch_pipe(
        ModuleType model, QueueType input_queue, QueueType output_queue,
        PipeType paramPipe);
//void make_predictions_torch_pipe();
void make_improvements_torch_pipe(
        ModuleType model, LearnQueueType in_queue, LossFunction loss_fn, int seq_length,
        OptimizerType optimizer, PipeType paramPipe, QueueType out_queue);

//void make_improvements_torch_pipe(
//        ModuleType model, LearnQueueType in_queue,LossFunction loss_fn, int seq_length,
//        OptimizerType optimizer, PipeType paramPipe);

// Make two one with one without out_queue
void push_data_to_queues(
        const torch::Tensor &data, int start, int endIndex,int time_skip,
        int input_size, QueueType currQueue, LearnQueueType histQueue);
void update_weights(
        ModuleType model, torch::OrderedDict<std::string, torch::Tensor> params,
        torch::OrderedDict<std::string, torch::Tensor> buffers);
void torch_update_state(ModuleType model, std::stringstream& new_state);
void push_data_to_pipes(torch::Tensor data, int start,
		int end_index, int time_skip, int input_size, int curr_pipe, int hist_pipe);

#endif // FORECASTER_H
