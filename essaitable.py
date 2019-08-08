import pandas as pd
import seaborn as sns
from flask import Flask, url_for, render_template, send_file, request

table = pd.read_csv(request.files.get("csv_file"))
graph = sns.scatterplot(x="colonne1", y="colonne2", data=table)
graph = graph.get_figure()
graph.savefig('static/images/output.png')