#include <iostream>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#if _MSC_VER
#include <Windows.h>
const auto closePipe = CloseHandle;
#else
#include <unistd.h>
const auto closePipe = close;
#endif

using std::vector;
using torch::optim::Adam;
using torch::optim::SGD;
using torch::nn::MSELoss;

typedef vector<at::Tensor> pipeBuffer;
typedef int QueueType;
typedef torch::jit::script::Module ModuleType;
typedef int PipeType;
//typedef at::Tensor (*loss_function)(at::Tensor, at::Tensor);
union LossFunction {
    MSELoss;
};
union OptimizerType {
    Adam;
    SGD;
};

void make_predictions_torch_pipe(
        ModuleType model, QueueType input_queue, QueueType output_queue,
        int *pipe);

void make_improvements_torch_pipe(
        ModuleType model, QueueType in_queue, int seq_length,
        OptimizerType optimizer, int *out_pipe);

void make_improvements_torch_pipe(
        ModuleType model, QueueType in_queue, LossFunction loss_fn, int seq_length,
        OptimizerType optimizer, int *out_pipe, QueueType out_queue);

void make_improvements_torch_pipe(
        ModuleType model, QueueType in_queue,LossFunction loss_fn, int seq_length,
        OptimizerType optimizer, int *out_pipe, QueueType out_queue);
// Make two one with one without out_queue
void torch_update_state(torch::jit::script::Module model, std::stringstream new_state);
void push_data_to_pipes(torch::Tensor data, int start,
		int end_index, int time_skip, int input_size, int curr_pipe, int hist_pipe);
