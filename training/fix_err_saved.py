import pandas as pd
import numpy as np
import models
import os

para = pd.read_csv("parafix.csv", sep=";")

custom_o = {f"mse_of_output_{i}": models.separate_mse(i) for i in range(6)}

for i, row in para.iterrows():
    np.save("n_res_block.npy", row["n_res_block"])
    for j in range(1, 6):
        filein = f"{row['in']}.{j}.keras"
        if os.path.exists(filein):
            loaded_model = keras.saving.load_model(
                filein,
                compile=True,
                custom_objects=custom_o,
            )
            loaded_model.save(filein)
