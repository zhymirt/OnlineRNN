#include <argp.h>
#include <toml.hpp>

class TorchRNN;
class TorchMLP;


load_data_numpy(filename: str)
load_data_pandas(filename: str)


save_outputs(
        predictions, actual, param_time, res_dir, metrics=None,
        metric_names=None, printed_metric_names=None)
get_args()
prepare_data(data, start_idx, pred_gap, slice_length)
