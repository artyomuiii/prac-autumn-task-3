from inspect import signature
import io
import time

import numpy as np
import pandas as pd

import plotly
import plotly.subplots
import plotly.graph_objects as go
from shapely.geometry.polygon import Point
from shapely.geometry.polygon import Polygon

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from random_forest import RandomForestMSE
from gradient_boosting import GradientBoostingMSE


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)


class RFParams:
    n_estimators = ''
    max_depth = ''
    feature_subsample_size = ''


class GBParams(RFParams):
    learning_rate = ''


rf_params = RFParams()
gb_params = GBParams()
data = None
data_val = None
data_test = None
data_shape = None
data_val_shape = None
data_test_shape = None
method_flag = 0


class RFParametersForm(FlaskForm):
    n_estimators = StringField('Кол-во деревьев', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина дерева', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля используемых признаков', validators=[DataRequired()])
    submit = SubmitField('Применить')


class GBParametersForm(FlaskForm):
    n_estimators = StringField('Кол-во деревьев', validators=[DataRequired()])
    learning_rate = StringField('Темп обучения', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина дерева', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля используемых признаков', validators=[DataRequired()])
    submit = SubmitField('Применить')


class FilesForm(FlaskForm):
    file_path_train = FileField('Тренировочные данные', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    '''file_path_val = FileField('Валидационные данные (опционально)', validators=[
        FileAllowed(['csv'], 'CSV only!')
    ])'''
    submit = SubmitField('Ввести')


class FilePredictForm(FlaskForm):
    file_path = FileField('Данные', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Ввести')


class PredictForm(FlaskForm):
    rmse = StringField('Итоговый RMSE на тренировочных данных', validators=[DataRequired()])
    fit_time = StringField('Время обучения модели, сек', validators=[DataRequired()])
    submit = SubmitField('Назад')


class ViewForm(FlaskForm):
    rmse = StringField('Итоговый RMSE на тренировочных данных', validators=[DataRequired()])
    rmse_val = StringField('Итоговый RMSE на валидационных данных', validators=[DataRequired()])
    fit_time = StringField('Время обучения модели, сек', validators=[DataRequired()])
    n_estimators = StringField('Кол-во деревьев', validators=[DataRequired()])
    learning_rate = StringField('Темп обучения (только для Gradient Boosting)', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина дерева', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля используемых признаков', validators=[DataRequired()])
    n_samples_train = StringField('Число объектов в тренировочных данных', validators=[DataRequired()])
    n_samples_test = StringField('Число объектов в валидационных данных (при наличии)', validators=[DataRequired()])
    n_features = StringField('Число признаков в данных', validators=[DataRequired()])
    submit = SubmitField('Назад')


def score_input(data, method_flag):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    X_train = data.drop(columns=['price', 'id', 'date']).values
    y_train = data.price.values
    print("method_flag", method_flag)
    if method_flag == 1:
        regressor = RandomForestMSE(rf_params.n_estimators, max_depth=rf_params.max_depth,
                                feature_subsample_size=rf_params.feature_subsample_size, random_state=42)
    elif method_flag == 2:
        regressor = GradientBoostingMSE(gb_params.n_estimators, learning_rate=gb_params.learning_rate,
                                        max_depth=gb_params.max_depth,
                                        feature_subsample_size=gb_params.feature_subsample_size, random_state=42)
    else:
        raise TypeError("score: undefined method!")

    start_time = time.time()
    regressor.fit(X_train, y_train)
    fit_time = time.time() - start_time

    print("fit_time", fit_time)

    y_train_pred = regressor.predict(X_train)
    rmse = mean_squared_error(y_train, y_train_pred, squared=False)

    return rmse, fit_time, regressor


def score_test(data_test, regressor):
    data_test['date'] = pd.to_datetime(data_test['date'])
    data_test['year'] = data_test['date'].dt.year
    data_test['month'] = data_test['date'].dt.month
    data_test['day'] = data_test['date'].dt.day
    # X_test = data_test.drop(columns=['id', 'date']).values
    X_test = data_test.drop(columns=['price', 'id', 'date']).values

    y_pred = regressor.predict(X_test)

    return y_pred


def score_val(data_val, regressor):
    data_val['date'] = pd.to_datetime(data_val['date'])
    data_val['year'] = data_val['date'].dt.year
    data_val['month'] = data_val['date'].dt.month
    data_val['day'] = data_val['date'].dt.day
    X_val = data_val.drop(columns=['price', 'id', 'date']).values
    y_val = data_val.price.values

    y_pred_val = regressor.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

    return rmse_val


@app.route('/')
@app.route('/index')
def index():
    global method_flag
    method_flag = 0
    return render_template('my_index.html', title='Выбор ML модели')


@app.route('/info')
def info():
    return render_template('my_info.html', title='Информация')


@app.route('/rand_forest_set_params', methods=['GET', 'POST'])
def rand_forest_set_params():
    rf_form = RFParametersForm()
    global method_flag
    method_flag = 1
    
    if rf_form.validate_on_submit():
        rf_params.n_estimators = int(rf_form.n_estimators.data)
        rf_params.max_depth = int(rf_form.max_depth.data)
        rf_params.feature_subsample_size = float(rf_form.feature_subsample_size.data)

        return redirect(url_for('files'))

    return render_template('my_from_form.html', form=rf_form, signature="Введите параметры")


@app.route('/grad_boost_set_params', methods=['GET', 'POST'])
def grad_boost_set_params():
    gb_form = GBParametersForm()
    global method_flag
    method_flag = 2

    if gb_form.validate_on_submit():
        gb_params.n_estimators = int(gb_form.n_estimators.data)
        gb_params.learning_rate = float(gb_form.learning_rate.data)
        gb_params.max_depth = int(gb_form.max_depth.data)
        gb_params.feature_subsample_size = float(gb_form.feature_subsample_size.data)
        
        return redirect(url_for('files'))

    return render_template('my_from_form.html', form=gb_form, signature="Введите параметры")


@app.route('/files', methods=['GET', 'POST'])
def files():
    files_form = FilesForm()

    if request.method == 'POST' and files_form.validate_on_submit():
        global data , data_val, data_shape, data_val_shape
        stream = io.StringIO(files_form.file_path_train.data.stream.read().decode("UTF8"), newline=None)
        data = pd.read_csv(stream)
        data_shape = data.shape
        '''stream_val = io.StringIO(files_form.file_path_val.data.stream.read().decode("UTF8"), newline=None)
        data_val = pd.read_csv(stream_val)
        data_val_shape = data_val.shape'''
        return redirect(url_for('result'))

    return render_template('my_files_form.html', form=files_form)


@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('my_result.html', title='Результат')


@app.route('/file_predict', methods=['GET', 'POST'])
def file_predict():
    file_predict_form = FilePredictForm()

    if request.method == 'POST' and file_predict_form.validate_on_submit():
        stream = io.StringIO(file_predict_form.file_path.data.stream.read().decode("UTF8"), newline=None)
        global data_test
        data_test = pd.read_csv(stream)
        return redirect(url_for('predict'))

    return render_template('my_file_predict_form.html', form=file_predict_form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predict_form = PredictForm()

    if predict_form.validate_on_submit():
        return redirect(url_for('result'))

    global data, method_flag
    rmse_train, fit_time, model = score_input(data, method_flag)
    y_pred = list(score_test(data_test, model))
    y_pred = ["%u" % x for x in y_pred]

    predict_form.rmse.data = "%.1u" % (rmse_train)
    predict_form.fit_time.data = "%.2f" % (fit_time)

    return render_template('my_from_form.html', form=predict_form, signature="Предсказание",
                            list=y_pred, name_list="Вектор предсказаний")


@app.route('/view', methods=['GET', 'POST'])
def view():
    view_form = ViewForm()

    if view_form.validate_on_submit():
        return redirect(url_for('result'))

    global data, data_val, data_shape, data_val_shape, method_flag
    rmse_train, fit_time, regressor = score_input(data, method_flag)

    view_form.rmse.data = "%.1u" % (rmse_train)
    view_form.fit_time.data = "%.2f" % (fit_time)
    view_form.n_samples_train.data = data_shape[0]
    view_form.n_features.data = data_shape[1]
    if data_val is not None:
        view_form.n_samples_test.data = data_val_shape[0]
        rmse_val = score_val(data_val, regressor)
        view_form.rmse_val.data = "%.1u" % (rmse_val)
    else:
        view_form.rmse_val.data = "-"
        view_form.n_samples_test.data = "-"
    
    if method_flag == 1:
        view_form.n_estimators.data = rf_params.n_estimators
        view_form.max_depth.data = rf_params.max_depth
        view_form.feature_subsample_size.data = rf_params.feature_subsample_size
        view_form.learning_rate.data = "-"
    elif method_flag == 2:
        view_form.n_estimators.data = gb_params.n_estimators
        view_form.learning_rate.data = gb_params.learning_rate
        view_form.max_depth.data = gb_params.max_depth
        view_form.feature_subsample_size.data = gb_params.feature_subsample_size
    else:
        raise TypeError("view: undefined method!")
    

    return render_template('my_from_form.html', form=view_form, signature="Информация о модели")