# ---------------------  Enregistrer une image ---------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.datasets import load_breast_cancer
def do_plot():
    # Loading

     # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    return bytes_image

# --------------------------------------------------------------------------------------