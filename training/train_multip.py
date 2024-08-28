#!/usr/bin/env python

import numpy as np
import gc
import keras
import os
import pickle
import time

import wandb.integration
import argparse
import wandb.integration.keras
import models
import wandb
import pandas as pd
from config import configs
from config import normalize_input_data
import os
import tensorflow as tf
import itertools as it
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbEvalCallback

os.environ["KERAS_BACKEND"] = "tensorflow"

project_name = "dbnets2.0.0"

__LABELS__ = [
    "Alpha",
    "AspectRatio",
    "InvStokes1",
    "FlaringIndex",
    "PlanetMass",
    "SigmaSlope",
]


# wandb callback for visualizing results
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(self, val_data_tuple, testing_resolutions=[0, 0.1, 0.15, 0.2]):
        super().__init__(
            ["idx", *[f"image_r{res}" for res in testing_resolutions], *__LABELS__],
            [
                "epoch",
                "idx",
                *[
                    f"{lab}_r{res}"
                    for res, lab in it.product(
                        testing_resolutions,
                        __LABELS__,
                    )
                ],
            ],
        )
        self.x = val_data_tuple[0]
        self.y = val_data_tuple[1]
        self.smoothed_x = {}
        self.testing_resolutions = testing_resolutions
        self.n = tf.shape(self.x)[0]

    def add_ground_truth(self, logs=None):
        # generating images with different smoothing
        for res in self.testing_resolutions:
            sigma = tf.ones(shape=(self.n, 1)) * res
            self.smoothed_x[f"{res}"] = self.model.get_smoothing_layer().smooth(
                self.x, sigma
            )
        for idx in range(self.n):
            images = [
                wandb.Image(self.smoothed_x[f"{res}"][idx])
                for res in self.testing_resolutions
            ]
            self.data_table.add_data(idx, *images, *self.y[idx])

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        all_preds = {}
        for res in self.testing_resolutions:
            all_preds[f"{res}"] = self._inference(res)

        table_idxs = self.data_table_ref.get_index()
        for idx in table_idxs:
            results = np.concatenate([
                all_preds[f"{res}"][idx] for res in self.testing_resolutions
            ])
            self.pred_table.add_data(epoch, self.data_table_ref.data[idx][0], *results)

    def _inference(self, res):
        # Compute predictions
        sigma = tf.ones(shape=(self.n, 1)) * res
        y_pred = self.model(self.smoothed_x[f"{res}"], res=sigma, training=False)
        return y_pred


# function that trains one fold
def train_core(params, data, fold):

    with wandb.init(project=project_name, config=params, name=params["name"]):

        # if running a sweep concatenate these parameters with those drawn by the agent
        if params["sweep"]:
            wandb.config.update(params)
            params = wandb.config

        # saving start time
        start_time = time.time()

        # loading the training data concatenating the different times
        print("Loading data")
        train_inp = np.concatenate(
            [
                np.expand_dims(data[f"time{t}"][f"inp_train{fold}"], axis=3)
                for t in params["times"]
            ],
            axis=0,
        )
        target_train = np.concatenate(
            [
                # this is necessary because labels have been wrongly packed
                np.concatenate(
                    [
                        data[f"time{t}"][f"targ_train{fold}"].reshape(-1, 6)[i::3]
                        for i in range(3)
                    ]
                )
                for t in params["times"]
            ],
            axis=0,
        )
        test_inp = np.concatenate(
            [
                np.expand_dims(data[f"time{t}"][f"inp_test{fold}"], axis=3)
                for t in params["times"]
            ],
            axis=0,
        )
        target_test = np.concatenate(
            [data[f"time{t}"][f"targ_test{fold}"] for t in params["times"]], axis=0
        )

        # normalizing input data
        print("Normalizing data")
        train_inp = normalize_input_data(train_inp)
        test_inp = normalize_input_data(test_inp)

        # instantiating the CNN model
        print("Creating CNN model")
        model = models.MultiPModel(
            act=params["activation"],
            dropout=params["dropout"],
            seed=params["seed"],
            maximum_translation_factor=params["maximum_translation_factor"],
            noise=params["noise"],
            maximum_res=params["maximum_augm_resolution"],
            training=True,
        )

        # preparing optimizer
        optimizer = keras.optimizers.Adam(learning_rate=params["lr"])

        # preparing callback for early stopping
        es2 = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=params["patience"],
            restore_best_weights=True,
        )
        if params["early_stopping"]:
            cb = [es2]
        else:
            cb = []

        # preparing metrics (mse computed on each output)
        metrics = [models.separate_mse(i) for i in range(6)]

        # preparing wandb callback
        wandb_callbacks = [
            WandbMetricsLogger(log_freq="epoch"),
            WandbClfEvalCallback(
                (
                    tf.convert_to_tensor(test_inp, dtype=tf.float32),
                    tf.convert_to_tensor(target_test, dtype=tf.float32),
                )
            ),
        ]
        cb = cb + wandb_callbacks

        # compiling model
        model.compile(loss="mse", optimizer=optimizer, metrics=metrics)

        # training
        print("Starting training")
        history = model.fit(
            x=train_inp,
            y=target_train,
            batch_size=params["batch_size"],
            epochs=500,
            verbose=1,
            shuffle=True,
            validation_data=(test_inp, target_test),
            callbacks=cb,
        )

        # computing summarizing scores after training
        print("Training finished. Computing evaluation metrics.")
        scores_test = model.evaluate(test_inp, target_test, verbose=0, return_dict=True)
        scores_train = model.evaluate(
            train_inp, target_train, verbose=0, return_dict=True
        )

        # save models
        print("Saving model")
        saving_file = f"{params['saving_folder']}/{params['name']}.{fold}.keras"

        model.save(saving_file)
        del model

        # save history
        with open(
            f"{params['saving_folder']}/histories/history.{fold}.hist", "wb"
        ) as file_h:
            pickle.dump(history.history, file_h)

        # save score
        with open(f"{params['saving_folder']}/scores.data", "a") as scores_file:
            # img size, fold, mse, mae
            scores_file.write(
                f"{params['name']},  {fold}, {scores_train['loss']}, {scores_test['loss']}, {time.time()-start_time}\n"
            )

        # save configuration
        params["wall_time"] = time.time() - start_time

        # log time to wandb
        wandb.log({"wall_time": params["wall_time"] / 1000})

        print(f'The process took {params["wall_time"]/1000} s.')
        del train_inp
        del test_inp
        gc.collect()


def train(params=None):

    if params is None:
        # it means we are running a sweep
        from sweep import global_params as params

        params["sweep"] = True
        params["saving_folder"] = (
            f"{params['saving_folder_g']}/{params['name']}/{time.time()}"
        )
    else:
        params["saving_folder"] = f"{params['saving_folder']}/{params['name']}"

    # checking if exists and creating output directory if it does not
    if os.path.exists(params["saving_folder"]):
        if params["override"]:
            print("Saving directory exists, overriding old data as instructed.")
        else:
            print(
                "WARNING! -> saving directory already exists, please run with Override=True"
            )
            exit()
    else:
        os.mkdir(params["saving_folder"])
        os.mkdir(f"{params['saving_folder']}/histories")

    # loading data
    data = {}
    for t in params["times"]:
        data[f"time{t}"] = np.load(
            f'{params["data_path"]}/{t}/data.npy', allow_pickle=True
        ).item()

    # checking file with parameter history and adding this run
    if os.path.exists("parahist.csv"):
        oldpara = pd.read_csv("parahist.csv", index_col=0)
        params["index"] = oldpara.index[-1] + 1
        newparafile = pd.concat([oldpara, pd.DataFrame([params]).set_index("index")])
    else:
        params["index"] = 0
        newparafile = pd.DataFrame([params]).set_index("index")
    newparafile.to_csv("parahist.csv")

    # begin train
    if params["resume"]:
        if not os.path.exists(params["resume_from"]):
            print(
                "Error! the model wich you want to resume from does not exist!\n Exiting..."
            )
            exit()
        else:
            # TODO: implement possibility to resume
            print("Error! Resuming not yet implemented")
            exit()
    else:
        for fold_no in [1, 2, 3, 4, 5]:
            params["fold"] = fold_no
            train_core(params, data, fold_no)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Trainer for dbnets2.0.0",
        description="Trains the CNN for dbnets2.0.",
    )

    parser.add_argument("-s", "--sweep", type=str, required=False, default=None)
    parser.add_argument("-ns", "--n-sweep-agents", required=False, default=1, type=int)

    args = parser.parse_args()

    if args.sweep is not None:
        print(f"Launching sweep agents performing {args.n_sweep_agents} iterations...")
        print(f"Sweep id: {args.sweep}")
        wandb.agent(args.sweep, train, count=args.n_sweep_agents, project=project_name)
    else:
        print("Starting training using parameters in configs.py...")
        for params in configs:
            train(params)
