import sys
from unittest import result
import keras
import numpy as np
import argparse
from matplotlib import testing
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import tensorflow as tf

# this makes the python files in ../training visible for import
sys.path.append("../training")

import models
from config import norm_functions
from train_multip import __LABELS__


def test(model, data, testing_resolutions=[0., 0.05, 0.1, 0.15], mcdrop=0, only_dropout=False):
    results = {}
    # Unpack the data
    x, y = data
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    results["loss"] = 0
    if mcdrop > 1:
        y = np.repeat(y, mcdrop, axis=0)
    for res in testing_resolutions:
        # generate convolved testing images
        sigma = tf.ones(shape=(tf.shape(x)[0], 1)) * res
        smoothed_x = model.get_smoothing_layer().smooth(x, sigma)
        results[f"smoothed_img_r{res}"] = smoothed_x
        if mcdrop > 1:
            sigma = np.repeat(sigma, mcdrop, axis=0)
            smoothed_x = np.repeat(smoothed_x, mcdrop, axis=0)
        # Compute predictions
        training = (mcdrop>1) and (not only_dropout)
        mcdropout = (mcdrop>1) or only_dropout
        y_pred = model(smoothed_x, res=sigma, training=training, no_smooth=True, mcdropout=mcdropout)
        results[f"y_pred_r{res}"] = y_pred.numpy().reshape(-1, mcdrop, 6)
        # Updates the metrics tracking the loss
        loss = model.compute_loss(y=y, y_pred=y_pred)
        # Update the metrics.
        for metric in model.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
                for name, val in metric.result().items():
                    results[f"{name}_r{res}"] = np.array([val])
            else:
                metric.update_state(loss)
                results[f"{metric.name}_r{res}"] = np.array([metric.result()])
                results["loss"] += metric.result() / len(testing_resolutions)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        for m in model.metrics:
            m.reset_state()
    results["loss"] = np.array([results["loss"]])
    return results


# settings through cli
parser = argparse.ArgumentParser(
    "Test data generator",
    description="generates predictions for all elements in the given test set",
)
parser.add_argument(
    "--model",
    "-m",
    required=True,
    type=str,
    help="path to keras model to be used for inference",
)
parser.add_argument(
    "--times",
    "-t",
    nargs="+",
    type=int,
    help="times of the snapshots to be used for testing",
    required=False,
    default=[500, 1000, 1500],
)
parser.add_argument("--data", "-d", help="path to data folder", type=str)
parser.add_argument("--inp-key", help="key to get inputs from the data dict", type=str, default='inp_test1')
parser.add_argument("--targ-key", help="key to get targets from the data dict", type=str,default='targ_test1')
parser.add_argument("--only-dropout", help='use only dropout for mc sampling of properties', action="store_true")
parser.add_argument(
    "--testing-resolutions",
    "-r",
    nargs="+",
    type=float,
    default=[0, 0.05, 0.1, 0.15],
    help="beam sizes in units of planet position to test for",
)
parser.add_argument(
    "--batch-size",
    "-b",
    help="size of batches of test images, reduce this number to reduce the memory usage in spite of a longer execution time",
    default=10,
    type=int
)
parser.add_argument(
    '--norm',
    '-n',
    type=str,
    help="name of the normalization function"
)
parser.add_argument('--fold', '-f', help='Id of fold')
parser.add_argument('--mc-drop',
                    '-mc',
                    help='number of samples for mc dropout',
                    default=1000,
                    required=False,
                    type=int)
parser.add_argument('--output', '-o', help='output file', type=str)
args = parser.parse_args()

custom_objs = {f"mse_of_output_{i}": models.separate_mse(i) for i in range(6)}
custom_objs['fold_no'] = models.get_fold_metric(args.fold)
# load model
loaded_model = keras.saving.load_model(
    args.model,
    compile=True,
    custom_objects={f"mse_of_output_{i}": models.separate_mse(i) for i in range(6)},
)

# load data
data = {}
for t in args.times:
    try:
        data[f"time{t}"] = np.load(f"{args.data}/{t}/data.npy", allow_pickle=True).item()
    except AttributeError:
        data[f"time{t}"] = np.load(f"{args.data}/{t}/data.npy", allow_pickle=True)

test_inp = np.concatenate(
    [np.expand_dims(data[f"time{t}"][args.inp_key], axis=3) for t in args.times],
    axis=0,
)
test_inp = norm_functions[args.norm](test_inp)

target_test = np.concatenate(
    [data[f"time{t}"][args.targ_key] for t in args.times], axis=0
)


# gen data and concatenate
n_batch = int(test_inp.shape[0] / args.batch_size)+1
results = {}
for i_batch in tqdm(range(n_batch), desc='Iterating over batches'):
    start_of_slice = i_batch*args.batch_size
    end_of_slice = (i_batch+1)*args.batch_size if i_batch<n_batch-1 else -1
    new_results = test(
        loaded_model,
        (test_inp[start_of_slice:end_of_slice], target_test[start_of_slice:end_of_slice]),
        testing_resolutions=args.testing_resolutions,
        mcdrop=args.mc_drop,
    )
    
    if i_batch==0:
        results = new_results
    else:
        for key in results.keys():
            results[key] = np.concatenate([results[key], new_results[key]], axis=0)

#save the dict results
print(f'I got the data, dumping it to {args.output}...')
with open(args.output, 'wb') as out_file:
    pickle.dump(results, out_file)
    
print('Done!')