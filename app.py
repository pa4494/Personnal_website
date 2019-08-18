from flask import Flask, url_for, render_template, send_file, request, redirect, jsonify, session, make_response
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from werkzeug import secure_filename
from pandas.api.types import is_numeric_dtype
import pandas as pd
import seaborn as sns
import os
from pathlib import Path
import inspect
from fastnumbers import fast_real
import pickle
import numpy as np
import joblib
import graphviz
import csv

app = Flask(__name__)
app.secret_key = "secret_key"
app.config["CACHE_TYPE"] = "null"
app.config['SESSION_TYPE'] = 'filesystem'

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
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dic_supervised_models = {"Linear_Regression" : LinearRegression,
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

dic_unsupervised_models = {"KMeans": KMeans,
                           "MiniBatchKMeans": MiniBatchKMeans,
                           "DBSCAN" : DBSCAN
                           }

list_label_encoder = ["Decision_Tree_Regressor",
                      "Random_Forest_Regressor",
                      "Decision_Tree_Classifier",
                      "Random_Forest_Classifier"]

list_dummy_drop_true = ["Linear_Regression",
                        "Logistic_Regression",
                        "Lasso_Regression",
                        "Ridge_Regression"]

list_dummy_drop_false =["Gaussian_Naive_bayes",
                        "Support_Vector_Machine",
                        "KMeans",
                        "MiniBatchKMeans",
                        "DBSCAN"]

list_norm = ["Logistic_Regression",
             "Lasso_Regression" ,
             "Ridge_Regression" ,
             "Support_Vector_Machine",
             "KMeans",
             "DBSCAN"]

list_lin_regr = ["Linear_Regression",
                 "Lasso_Regression",
                 "Ridge_Regression"]

list_classif = ["Logistic_Regression",
                "Decision_Tree_Classifier",
                "Random_Forest_Classifier",
                "Gaussian_Naive_bayes",
                "Support_Vector_Machine"]

list_tree = ["Decision_Tree_Classifier",
             "Decision_Tree_Regressor"]

list_KMeans = ["KMeans",
               "MiniBatchKMeans"]

# ---------  Unsupervised ML models --------------



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
            plt.close("all")

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
            plt.close("all")

        response = make_response(render_template("bivariate.html", dataset = dataset, nb_options = nb_options, table_head = table_head, dic_graphs = dic_graphs, dic_func= dic_func, nb_graphs = nb_graphs, code = code))
        return response

# based
@app.route('/models/<type_model>/', methods= ["GET", "POST"])
def model_selection(type_model):

    code =""
    table_head=""
    template = ""
    has_null = False
    dic_models = {}

    # If no csv file has been uploaded, we redirect to the upload page
    if not session.get("file_path"):
        return redirect(url_for("upload"))

    # If user tries to manually enter a URL /models/TEXT_TYPED/ which is not accepted, we redirect to the upload page
    if type_model not in ["supervised_ML", "unsupervised_ML"]:
        return redirect(url_for("upload"))

    else:
        session["type_model"] = type_model
        dataset = pd.read_csv(session["file_path"])
        detail_null = dataset.isnull().sum()
        if sum(detail_null) > 0:
            has_null =True

        code+="<p> dataset = pd.read_csv({})</p>".format(Path(session["file_path"]).name)
        table_head = dataset.head(10).to_html()

    if type_model == "supervised_ML":
        dic_models = dic_supervised_models
    if type_model == "unsupervised_ML":
        dic_models = dic_unsupervised_models

    return render_template("model_selection.html", type_model = type_model, dataset = dataset, table_head = table_head, has_null = has_null, detail_null = detail_null, dic_models = dic_models)

@app.route("/params", methods = ["POST"])
def model_parameters():

    session["model"] = request.form.get("model")
    IP_address_user= request.remote_addr
    dataset = pd.read_csv(session["file_path"])
    help_model = ""
    session["le_y"] = ""
    session["le_X"] = {}
    X_train = ""
    X_test = ""
    y_train = ""
    y_test = ""
    dic_models = {}
    stratify = None
    silhouette_score=[]

    # We delete columns based on user's choice
    for column in request.form.getlist("delete"):
        dataset.drop(column, axis=1, inplace=True)

    # We treat null values
    if request.form.get("treat_null") == "remove_rows":
        dataset.dropna(axis=0, inplace=True)
    elif request.form.get("treat_null") == "interpolate":
        dataset.interpolate(method="nearest", inplace=True)

    # If model_type is "supervised", we split dataset between X and y and make sure y is numerical. Else X = dataset
    if session["type_model"] == "supervised_ML" :

        dic_models = dic_supervised_models
        y = pd.DataFrame(dataset[request.form.get("target")], columns= [request.form.get("target")])
        X = dataset.drop(request.form.get("target"), axis=1)

        # We make sure target is expressed as a number
        if not is_numeric_dtype(y.iloc[:,0]):
            le = LabelEncoder()
            y = pd.DataFrame(le.fit_transform(y.astype(str)), columns=y.columns)
            path = os.path.join(uploads_dir, "Label_encoder_y" + IP_address_user + ".data")
            with open(path, 'wb') as filehandle:
                # store the data as binary data stream
                pickle.dump(list(le.classes_), filehandle)
                session["le_y_path"] = path

    if session["type_model"] == "unsupervised_ML":
        dic_models = dic_unsupervised_models
        X = dataset

    help_model = dic_models[session["model"]].__doc__.split('\n')
    # We dummify categorical variables or label encode them based on model used
    if request.form.get("model") in list_dummy_drop_true:
        X = pd.get_dummies(X, drop_first=True)

    if request.form.get("model") in list_dummy_drop_false:
        X =pd.get_dummies(X, drop_first=False)

    if request.form.get("model") in list_label_encoder:
        for column in X.columns:
            if not is_numeric_dtype(X[column]):
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
                path = os.path.join(uploads_dir, "Label_encoder_" + column + IP_address_user + ".data")
                with open(path, 'wb') as filehandle:
                    # store the data as binary data stream
                    pickle.dump(list(le.classes_), filehandle)
                    session["le_X"][column] = path

    # If model_type is "supervised", we split the dataset between train and test, we normalise based on model chosen and we save X and y train and test in csv files
    # If model type is "unsupervised", we normalise based on model chosen and we save pre_processed X as a csv file

    if session["type_model"] == "supervised_ML":
        from sklearn.model_selection import train_test_split
        # if y has at least one value which is unique, we do not stratify
        if y.iloc[:, 0].value_counts().min() >= 2:
           stratify = y
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=42,
                                                            stratify=stratify)  # ATTENTION. IF y has at least one class with a unique entry this will return an error
        # We normalise when needed
        if request.form.get("model") in list_norm:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns= X_train.columns)
            X_test = pd.DataFrame(sc_X.transform(X_test), columns = X_test.columns)

        # We save a version of X and y with user IP address in each file name (to allow several users to use the tool at the same time)
        for data, name in [(X_train, "X_train_"), (X_test, "X_test_"), (y_train, "y_train_"),
                           (y_test, "y_test_")]:
            path = os.path.join(uploads_dir, name + IP_address_user +".csv")
            data.to_csv(path)
            session[name+"path"] = path

    if session["type_model"] == "unsupervised_ML":
        # We normalise when needed
        if request.form.get("model") in list_norm:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            X = pd.DataFrame(sc_X.fit_transform(X), columns= X.columns)
            # We save the model to de-normalize before putting on graphs
            path_std_scaler = os.path.join("static/models/", "normalisation_model_" + request.remote_addr + ".pkl")
            joblib.dump(sc_X, path_std_scaler)
            session["path_std_scaler"] = path_std_scaler

        # we save a version of X as X_unsupervised.csv
        path = os.path.join(uploads_dir, f"X_unsupervised_{IP_address_user}.csv")
        X.to_csv(path)
        session["X_unsupervised_path"] = path

        # if model is part of the KMeans family we generate an elbow graph and a silhouette analysis
        if session["model"] in list_KMeans:
            from KMeans_graphs import elbow_graph
            elbow_graph = elbow_graph(X)
            elbow_graph.savefig("static/images/elbow_graph.png")
            plt.close("all")

            from KMeans_graphs import silhouette
            silhouette, silhouette_score = silhouette(X, X_axis = request.form.get("X_axis"), y_axis = request.form.get("y_axis"))
            i = 0
            for figure in silhouette:
                figure.savefig(f"static/images/silhouette_{i}.png")
                i += 1
            plt.close("all")

    # We get a list of all the parameters taken by the model chosen by user
    signature = inspect.getfullargspec(dic_models[session["model"]])
    # signature.args include self as first element whereas signature.defaults does not. Hence taking [::-1] takes every pair defacto excluding self
    list_params = list(zip(signature.args[::-1], signature.defaults[::-1]))
    list_params.sort()
    return render_template("choice_model_parameters.html", list_params= list_params, help_model= help_model, silhouette_score=silhouette_score)

@app.route("/result_model", methods = ["GET","POST"])
def result_model():

    dic_models = {}
    report = {}
    conf_matrix = ""
    is_classifier = False
    is_lin_regr = False
    is_tree = False
    df_coefs = ""
    df_feat_rank = ""
    X_train = pd.read_csv(session["X_train_path"], index_col = 0)
    X_test = pd.read_csv(session["X_test_path"], index_col = 0)
    y_train = pd.read_csv(session["y_train_path"], index_col = 0)
    y_test = pd.read_csv(session["y_test_path"], index_col = 0)
    X_unsupervised = pd.read_csv(session["X_unsupervised_path"], index_col=0)
    nb_classes_y = len(y_train.iloc[:, 0].unique())
    if session["type_model"] == "supervised_ML":
        dic_models = dic_supervised_models
    elif session["type_model"] == "unsupervised_ML":
        dic_models = dic_unsupervised_models
    # We get the parameters and default value for all parameters of the model chosed by user
    signature = inspect.getfullargspec(dic_models[session["model"]])
    dic_params = {}

    # We take all elements of signature except first one which is "self"
    for arg in signature.args[1:]:
        if request.form.get(arg) in ["True", "False", "None"]:
            dic_params[arg] = eval(request.form.get(arg))
        else :
            # fast_real intelligently converts text into text, int or float,
            dic_params[arg] = fast_real(request.form.get(arg))

    model = dic_models[session["model"]](**dic_params)

    if session["type_model"] == "supervised_ML":

        model.fit(X_train, y_train)

        #  ------------------   Visualisation / model evaluation --------------------

        # In the case where user picked a classification model we generate classification reports for train and test along with a confusion matrix
        # If y was not numerical, we get back label names from pickle file

        if session["model"] in list_classif:
            is_classifier = True
            if not is_numeric_dtype(y_train.iloc[:, 0]):
                with open(session["le_y_path"], 'rb') as filehandle:
                    le_y = pickle.load(filehandle)
            else :
                le_y = None

            for X, y, name in [(X_train, y_train, "train"), (X_test, y_test, "test")]:

                # We create a classification report for train and test
                report[name] = classification_report(y_true= y , y_pred= model.predict(X), target_names= le_y, output_dict=True)
                report[name] = pd.DataFrame(report[name]).to_html()

                # We create a confusion matrix for train and test
                conf_matrix = pd.DataFrame(confusion_matrix(y_true= y, y_pred=model.predict(X)))
                akws = {"ha": 'center', "va": "center"}
                graph_train = sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws=akws)
                plt.title(name)
                graph_train.figure.savefig(f"static/images/confusion_matrix_{name}.png", bbox_inches="tight")
                plt.close("all")

            # if y is binary, we create a ROC curve graph
            if nb_classes_y == 2 :
                from sklearn.metrics import roc_auc_score
                from sklearn.metrics import roc_curve

                for X, y, name in [(X_train, y_train,"train"), (X_test, y_test, "test")] :
                    # Roc curve on test
                    logit_roc_auc = roc_auc_score(y, model.predict(X))
                    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
                    plt.figure()
                    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
                    plt.plot([0, 1], [0, 1], 'r--')
                    plt.xlim([-0.05, 1.05])
                    plt.ylim([-0.05, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'Roc curve {name}')
                    plt.legend(loc="lower right")
                    plt.savefig(f"static/images/roc_{name}.png", bbox_inches="tight")
                    plt.close("all")

        # If model was a linear regressor :
        if session["model"] in list_lin_regr:
            is_lin_regr = True

            # We show coefficient of the model associated to each feature
            df_coefs = pd.DataFrame(index=X_train.columns, data=model.coef_.transpose(), columns=["coefficients"])
            df_coefs = df_coefs.sort_values(by = "coefficients", ascending = False).to_html()

            # We show f-score and p-value of each feature
            from sklearn.feature_selection import f_regression
            feature_importance = f_regression(X_train, y_train)
            df_feat_rank = pd.DataFrame(columns=X_train.columns, data=feature_importance,
                                        index=["f-score", "p-value"]).transpose().sort_values(["f-score", "p-value"], ascending=False)
            df_feat_rank = df_feat_rank.to_html()

        # if model is a tree we generate a png representation of that tree
        if session["model"] in list_tree:
            is_tree = True

            from sklearn.tree import export_graphviz
            from subprocess import check_call

            graph_tree = export_graphviz(model, out_file="static/images/tree.dot", node_ids="box", feature_names=X_train.columns)
            check_call(['dot', '-Tpng', "static/images/tree.dot", '-o', 'static/images/tree.png'], shell=True)

    # if model type is "unsupervised" generate a graph for all combination of 2 features and show in which cluster they belong

    if session["type_model"] == "unsupervised_ML" :
        model = dic_models[session["model"]](**dic_params)
        model.fit(X_unsupervised)
        # for better clarity before showing clusters on graph we de-normalise the dataset
        if session["model"] in list_norm:
            #with open(session["path_std_scaler"], 'wb') as filehandle:
            std_scaler = joblib.load(session["path_std_scaler"])
            X_unsupervised = pd.DataFrame(std_scaler.inverse_transform(X_unsupervised), columns=X_unsupervised.columns)
            path = os.path.join(uploads_dir, f"check=konX.csv")
            #X.to_csv(path)

        nb_graphs = int(1/2*len(X_unsupervised.columns)*(len(X_unsupervised.columns)-1))
        nb_columns = 6
        nb_rows = int(nb_graphs / nb_columns - 0.0000001) + 1
        width_fig = 24
        fig, axis = plt.subplots(nrows=nb_rows, ncols=nb_columns, figsize=(width_fig, width_fig / nb_columns * nb_rows))
        ax = list(axis.flat)
        for x in range(len(X_unsupervised.columns)-1):
            for y in np.arange(x+1,len(X_unsupervised.columns)):
                nb_graphs-=1
                sns.scatterplot(x=X_unsupervised.iloc[:,x], y=X_unsupervised.iloc[:,y], data=X_unsupervised, hue=model.labels_, ax=ax[nb_graphs], legend = "full")
                # if model is part of the KMeans group we plot centroids as well
                if session["model"] in list_KMeans:
                    centers = model.cluster_centers_
                    if session["model"] in list_norm:
                        centers = std_scaler.inverse_transform(centers)
                    ax[nb_graphs].scatter(centers[:,x], centers[:,y], marker='o', c="white", alpha=1, s=200, edgecolor='k')
                    for i, c in enumerate(centers):
                        ax[nb_graphs].scatter(c[x], c[y], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

                   # coords_centers = pd.DataFrame(coords_centers, columns=X_unsupervised.columns)
                    #sns.scatterplot(x=coords_centers.iloc[:,x], y=coords_centers.iloc[:,y], data=coords_centers, s=80, ax=ax[nb_graphs], legend="full", label="centroids",  )
        fig.savefig("static/images/clusters.png", bbox_inches="tight")
        plt.close("all")

    # Finally we export the model using pickle to allow user to download it, using IP address of user in the name of the file

    path_pickle = os.path.join("static/models/", session["model"] + request.remote_addr + ".pkl")
    joblib.dump(model, path_pickle)

    return render_template("result_model.html", model = model, X_train = X_train, X_test = X_test, y_train= y_train, y_test = y_test, path_pickle = path_pickle,
                           dic_params=dic_params, report= report, is_classifier=is_classifier, is_lin_regr= is_lin_regr, nb_classes_y=nb_classes_y, is_tree = is_tree,
                           df_coefs=df_coefs, df_feat_rank= df_feat_rank)

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









