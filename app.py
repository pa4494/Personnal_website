from typing import Dict, Union

from flask import Flask, url_for, render_template, send_file, request, redirect, jsonify, session, make_response
from werkzeug import secure_filename
from pandas.api.types import is_numeric_dtype
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import inspect
import pprint
import csv

app = Flask(__name__)
app.secret_key = "secret_key"
app.config["CACHE_TYPE"] = "null"

# Folder where the csv files uploaded by users will be stored
uploads_dir = os.path.join(app.instance_path, 'uploads')

UPLOAD_FOLDER = "static/images"
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dic_models = {"Linear_Regression" : LinearRegression,
              "Logistic_Regression" : LogisticRegression,
              "Lasso_Regression" : Lasso,
              "Ridge_Regression" : Ridge,
              "Decision_Tree_Regressor" : DecisionTreeRegressor,
              "Random_Forest_Regressor": RandomForestRegressor,
              "Decision_Tree_Classifier" : DecisionTreeClassifier,
              "Random_Forest_Classifier" : RandomForestClassifier,
              "Gaussian_Naive_bayes" : GaussianNB,
              "Support_Vector_Machine" : SVC
              }

list_dummy = ["Linear_Regression",
              "Logistic_Regression",
              "Lasso_Regression" ,
              "Ridge_Regression" ,
              "Gaussian_Naive_bayes" ,
              "Support_Vector_Machine"]

list_norm = ["Logistic_Regression",
             "Lasso_Regression" ,
             "Ridge_Regression" ,
             "Decision_Tree" ,
             "Gaussian_Naive_bayes" ,
             "Support_Vector_Machine"]

list_regr = ["Linear_Regression",
             "Logistic_Regression",
             "Lasso_Regression",
             "Ridge_Regression",
             "Decision_Tree_Regressor",
             "Random_Forest_Regressor"]

list_classif = ["Decision_Tree_Classifier",
                "Random_Forest_Classifier",
                "Gaussian_Naive_bayes",
                "Support_Vector_Machine"]

# Functions we want to use in html template using jinja2
@app.context_processor
def utility_processor():
    def name_file(file_path):
        return Path(file_path).stem
    return dict(name_file=name_file, is_numeric= is_numeric_dtype, len = len, min = min, str=str, type = type)

# Indicates we do not want cache memory (Graphs are saved and shown as images. If not set, web app does not update graphs after form submission)
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
                        dic_func[key](dataset[column].dropna(), ax=ax[nb_graphs])
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
    x_list =[]
    y_list =[]
    code = "<p> # Import all relevant libraries </p>"

    # Dictionnary containing graph functions associated with drop-down list
    dic_func = {"scatterplot": sns.scatterplot,
                "barplot": sns.barplot}

    # If no csv file has been uploaded, we redirect to the upload page
    if not session.get("file_path"):
        return redirect(url_for("upload"))
    else :
        dataset = pd.read_csv(session["file_path"])
        code+="<p> dataset = pd.read_csv({})</p>".format(Path(session["file_path"]).name)
        table_head = dataset.head(10).to_html()
        nb_options = min(7, len(dataset.columns))

        if request.method == "POST":

            # After user submits button radio form with choice of variable to plot on graph, each of the radio button's value becomes a dictionary key and to each key is associated
            # a list of the column names the user wants to plot
            for var_nb in range(nb_options):
                var_x = "var_x_" + str(var_nb)
                var_y = "var_y_" + str(var_nb)
                choice_x = request.form.get(var_x)
                choice_y = request.form.get(var_y)
                if choice_x != "":
                    x_list.append(choice_x)
                if choice_y != "":
                    y_list.append(choice_y)

            # We create a subplots figures that automatically resize based on the number of graphs we want to show
            nb_columns = 3
            graphs_to_show = len(x_list) * len(y_list)
            nb_rows = int(graphs_to_show/nb_columns -0.0000001) + 1
            width_fig = 12
            f, axis = plt.subplots(nrows = nb_rows, ncols = nb_columns, figsize=(width_fig, width_fig/nb_columns*nb_rows))
            ax = list(axis.flat)

            # For all the columns that the user wants to see a graph, we plot a graph based on the radio button's value
            graph_choice = dic_func[request.form.get("graph_type")]

            for x in x_list :
                for y in y_list:
                    graph_choice(x = x, y=y, data=dataset.dropna(), ax=ax[nb_graphs])
                    nb_graphs+=1
                    code+="<P> sns.{graph_choice}(x = {x}, y={y}, data=dataset]))</P>".format(graph_choice = graph_choice.__name__, x=x, y=y)

            # We save the figure containing all the graph
            f.savefig('static/images/graph_bivariate.png')

        response = make_response(render_template("bivariate.html", dataset = dataset, nb_options = nb_options, table_head = table_head, dic_graphs = dic_graphs, dic_func= dic_func, nb_graphs = nb_graphs, code = code))
        return response

@app.route('/supervised_ML/', methods = ["GET", "POST"])
def supervised_ML():

    code =""
    table_head=""
    has_null = False

    # If no csv file has been uploaded, we redirect to the upload page
    if not session.get("file_path"):
        return redirect(url_for("upload"))

    else:
        dataset = pd.read_csv(session["file_path"])
        detail_null = dataset.isnull().sum()
        if sum(detail_null) > 0:
            has_null =True

        code+="<p> dataset = pd.read_csv({})</p>".format(Path(session["file_path"]).name)
        table_head = dataset.head(10).to_html()

    return render_template("supervised_ML.html", dataset = dataset, table_head = table_head, has_null = has_null, detail_null = detail_null, dic_models = dic_models)

@app.route("/params", methods = ["POST"])
def model_parameters():

    session["model"] = request.form.get("model")
    IP_address_user_= "A_DEFINIR"
    dataset = pd.read_csv(session["file_path"])
    help_model = dic_models[session["model"]].__doc__.split('\n')
    # We delete columns based on user's choice
    for column in request.form.getlist("delete"):
        dataset.drop(column, axis=1, inplace=True)

    # We treat null values
    if request.form.get("treat_null") == "remove_rows":
        dataset.dropna(axis=0, inplace=True)
    else:
        dataset.interpolate(method="nearest", inplace=True)

    # We make sure target is expressed as a number
    if not is_numeric_dtype(dataset[request.form.get("target")]):
        print("APPLY ONE HOT ENCODER AND KEEP TABLE OF CORRESPONDING VALUE IN A TABLE")

    # We dummify categorical variables
    if request.form.get("model") in list_dummy:
        dataset = pd.get_dummies(dataset, drop_first=True)

    # We split dataset between X and y
    y = dataset[request.form.get("target")]
    X = dataset.drop(request.form.get("target"), axis=1)

    # We split the dataset between train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y)  # ATTENTION. IF y has at least one class with a unique entry this will return an error
    # We normalise
    if request.form.get("model") in list_norm:
        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns= X_train.columns)
        X_test = pd.DataFrame(sc_X.transform(X_test), columns = X_test.columns)

    # We save a version of X and y with user IP address in each file name (to allow several users to use the tool at the same time)
    for data, name in [(X_train, "X_train"), (X_test, "X_test"), (y_train, "y_train"),
                       (y_test, "y_test")]:
        path = os.path.join(uploads_dir, name + IP_address_user_ +".csv")
        # To cover the case where data is a numpy, we apply pd.DataFrame
        data.to_csv(path) # NEED TO ADD THE CODE TO GET THE IP ADDRESS
        session[name+"_path"] = path

    # We get a list of all the parameters taken by the model chosen by user
    signature = inspect.getfullargspec(dic_models[session["model"]])
    list_params = list(zip(signature.args[::-1], signature.defaults[::-1]))
    list_params.sort()

    return render_template("choice_model_parameters.html", list_params= list_params, help_model= help_model)

@app.route("/result_model", methods = ["POST"])
def result_model():
    X_train = pd.read_csv(session["X_train_path"])
    X_test = pd.read_csv(session["X_test_path"])
    y_train = pd.read_csv(session["y_train_path"], names=["target"])
    y_test = pd.read_csv(session["y_test_path"], names=["target"])

    listo = request.form.getlist("params")
    #for elem in request.form.getlist("params"):

    signature = inspect.getfullargspec(dic_models[session["model"]])
    dic_params = {}
    for arg in signature.args[1:]:
        dic_params[arg] = request.form.get(arg)

    model = dic_models[session["model"]](**dic_params)
    model.fit(X_train,y_train)

    return render_template("result_model.html", model = model, X_train = X_train, X_test = X_test, y_train= y_train, y_test = y_test, dic_params=dic_params)

@app.route("/test_html", methods = ["POST"])
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









