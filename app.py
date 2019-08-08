from flask import Flask, url_for, render_template, send_file, request, redirect, jsonify, session, make_response
from werkzeug import secure_filename
from pandas.api.types import is_numeric_dtype
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import csv

app = Flask(__name__)
app.secret_key = "secret_key"
app.config["CACHE_TYPE"] = "null"

# Folder where the csv files uploaded by users will be stored
uploads_dir = os.path.join(app.instance_path, 'uploads')

UPLOAD_FOLDER = "static/images"
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Functions we want to use in html template using jinja2
@app.context_processor
def utility_processor():
    def name_file(file_path):
        return Path(file_path).stem
    return dict(name_file=name_file, is_numeric= is_numeric_dtype, len = len, min = min)

# Indicate we do not want cache memory (Graphs are saved and showns as images. If not set, web app does not update graphs after form submission)
@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# Page the user is redirected to if no csv file has been uploaded and he is trying to access analysis pages
@app.route("/upload", methods = ["GET", "POST"])
def upload() :
# Page that allows user to upload data. If data has already been loaded for the session, the page shows the data and the file name

    table_head = {}
    if request.method == "POST":
        # We save the file and allocate file path to file_path session variable
        data_file = request.files['csv_file']
        session["file_path"] = os.path.join(uploads_dir, secure_filename(data_file.filename))
        data_file.save(session["file_path"])

    # If a file_path session variable already exists, we take head(10) and convert it to html to then show it on the page
    if session.get("file_path"):
        table_head = pd.read_csv(session["file_path"]).head(10).to_html()
    return render_template("upload.html", table_head = table_head)

# page to allow user to create univariate plots from the csv file
@app.route("/univariate_exploratory", methods = ["GET", "POST"])
def univariate():
    dic_graphs = {}
    #dic_types = {}
    nb_graphs = 0
    code = "<p> # Import all relevant libraries </p>"

    # If no csv file has been uploaded, we redirect to the upload page
    if not session.get("file_path"):
        return redirect(url_for("upload"))
    else :
        dataset = pd.read_csv(session["file_path"])
        code+="<p> dataset = pd.read_csv({})</p>".format(Path(session["file_path"]).name)
        table_head = dataset.head(10).to_html()

        if request.method == "POST":

            # Dictionnary containing graph functions associated with radio buttons
            dic_func = {"distplot" : sns.distplot,
                        "countplot" : sns.countplot}

            # After user submits button radio form with choice of variable to plot on graph, each of the radio button's value becomes a dictionary key and to each key is associated
            # a list of the column names the user wants to plot
            for column in dataset.columns:
                dic_graphs[request.form[column]] = dic_graphs.get(request.form[column], []) + [column]

            # We create a subplots figures that automatically resize based on the number of graphs we want to show
            nb_columns = 3
            graphs_to_show = len(dataset.columns) - len(dic_graphs.get("none",[]))
            nb_rows = int(graphs_to_show/nb_columns -0.0000001) + 1
            width_fig = 14
            f, axis = plt.subplots(nrows = nb_rows, ncols = nb_columns, figsize=(width_fig, width_fig/nb_columns*nb_rows))
            ax = list(axis.flat)

            # For all the columns that the user wants to see a graph, we plot a graph based on the radio button's value
            for key in dic_graphs.keys():
                if key != "none":
                    for column in dic_graphs.get(key):
                        dic_func[key](dataset[column], ax=ax[nb_graphs])
                        nb_graphs+=1
                        code+="<P> sns.{}(dataset[{}]))</P>".format(dic_func[key].__name__,column)

            # We save the figure containing all the graph
            f.savefig('static/images/graph_univariate.png')

        response = make_response(render_template("univariate.html", dataset = dataset, table_head = table_head, dic_graphs = dic_graphs, nb_graphs = nb_graphs, code = code))
        return response

# Page to allow user to create bivariate plots from csv file. Can choose several x-axis variables and y-axis variables and choose a graph type.
# The algorithm will then create for each x a graph with all y, with graph type chosen by user
@app.route("/bivariate_exploratory", methods = ["GET", "POST"])
def bivariate():
    dic_graphs = {}
    nb_graphs = 0
    nb_options = 0
    code = "<p> # Import all relevant libraries </p>"

    # If no csv file has been uploaded, we redirect to the upload page
    if not session.get("file_path"):
        return redirect(url_for("upload"))
    else :
        dataset = pd.read_csv(session["file_path"])
        code+="<p> dataset = pd.read_csv({})</p>".format(Path(session["file_path"]).name)
        table_head = dataset.head(10).to_html()
        nb_options = min(7, len(dataset.columns))

        if request.method == "POST":

            # Dictionnary containing graph functions associated with radio buttons
            dic_func = {"distplot" : sns.distplot,
                        "countplot" : sns.countplot}

            # After user submits button radio form with choice of variable to plot on graph, each of the radio button's value becomes a dictionary key and to each key is associated
            # a list of the column names the user wants to plot
            for column in dataset.columns:
                dic_graphs[request.form[column]] = dic_graphs.get(request.form[column], []) + [column]

            # We create a subplots figures that automatically resize based on the number of graphs we want to show
            nb_columns = 3
            graphs_to_show = len(dataset.columns) - len(dic_graphs.get("none",[]))
            nb_rows = int(graphs_to_show/nb_columns -0.0000001) + 1
            width_fig = 12
            f, axis = plt.subplots(nrows = nb_rows, ncols = nb_columns, figsize=(width_fig, width_fig/nb_columns*nb_rows))
            ax = list(axis.flat)

            # For all the columns that the user wants to see a graph, we plot a graph based on the radio button's value
            for key in dic_graphs.keys():
                if key != "none":
                    for column in dic_graphs.get(key):
                        dic_func[key](dataset[column], ax=ax[nb_graphs])
                        nb_graphs+=1
                        code+="<P> sns.{}(dataset[{}]))</P>".format(dic_func[key].__name__,column)

            # We save the figure containing all the graph
            f.savefig('static/images/graph_bivariate.png')

        response = make_response(render_template("bivariate.html", dataset = dataset, nb_options = nb_options, table_head = table_head, dic_graphs = dic_graphs, nb_graphs = nb_graphs, code = code))
        return response

@app.route('/supervised_ML', methods = ["GET", "POST"])
def supervised_ML():
    variable = pd.read_csv(session["file_path"])
    return render_template("supervised_ML.html", name = "Supervised Machine Learning", variable = variable)

@app.route("/test_html", methods = ["GET", "POST"])
def test_html():
    var = "on est dans get"
    if request.method == "POST":
        if "choose" in request.form :
            var = "vous avez choisi choose"
        else:
            var = "vous avez uploade"

    return render_template('test_html.html', var = var)

if __name__ == '__main__':
    app.run(debug=True)









