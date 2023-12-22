from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
from pytz import timezone
import joblib
import os
import pandas as pd
from model.model_KNN import KNN
from model.model_Decision_Tree import DecisionTree
from model.model_Naive_Bayes import NaiveBayes
from model.model_SVM import SVM
import re
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

# Model for Prediction History
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    features = db.Column(db.String(255))
    model = db.Column(db.String(30))
    result = db.Column(db.String(20))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def winsorize(data, attribute, alpha=0.05):
    sorted_data = data.sort_values(by=attribute)
    length = len(sorted_data)

    lower_cutoff_index = int(alpha / 2 * length)
    upper_cutoff_index = int((1 - alpha / 2) * length)

    lower_bound = sorted_data[attribute].iloc[lower_cutoff_index]
    upper_bound = sorted_data[attribute].iloc[upper_cutoff_index]

    winsorized_data = data.copy()

    winsorized_data[attribute] = np.where(winsorized_data[attribute] < lower_bound, lower_bound, np.where(winsorized_data[attribute] > upper_bound, upper_bound, winsorized_data[attribute]))

    return winsorized_data


csv_file_path = 'data\data_credit_card.csv'
credit_card_data = pd.read_csv(csv_file_path)
credit_card_data = winsorize(credit_card_data,"Amount", 0.1)
scaler1 = StandardScaler()

credit_card_data['Vamount'] = scaler1.fit_transform(credit_card_data["Amount"].values.reshape(-1,1))

credit_card_data = credit_card_data.drop(["Amount"], axis = 1)

X_ori = credit_card_data.drop(["Class"], axis = 1)
y_ori = credit_card_data["Class"]

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n = 492, random_state = 421)

new_dataset = pd.concat([legit_sample, fraud], axis = 0)

new_dataset = winsorize(new_dataset,'Vamount', 0.15)

X = new_dataset.drop(["Class"], axis = 1)
y = new_dataset["Class"]

index = X.index
pca = PCA(n_components = 5)
principalComponents = pca.fit_transform(X)
principalDf  = pd.DataFrame(data = principalComponents, columns = ["PC1", "PC2","PC3","PC4","PC5"], index = index)

finalDf = pd.concat([principalDf, y], axis = 1)

X1 = finalDf.drop(columns="Class", axis = 1)
Y1 = finalDf["Class"]

X = X1.to_numpy()
y = Y1.to_numpy()
scaler2 = StandardScaler()
X = scaler2.fit_transform(X)

def model_file_exists(filename):
    return os.path.isfile(filename)

# Function to train and save the model
def train_and_save_model(model, model_filename):
    model.fit(X, y)

    # Lưu mô hình vào một file .joblib
    joblib.dump(model, model_filename)

    return model

# Đường dẫn cho các file .joblib
knn_model_filename = 'knn_model.joblib'
decision_tree_model_filename = 'decision_tree_model.joblib'
naive_bayes_model_filename = 'naive_bayes_model.joblib'
svm_model_filename = 'svm_model.joblib'

# Kiểm tra và load model nếu nó tồn tại, nếu không thì huấn luyện và lưu
def load_or_train_model(model, filename):
    if model_file_exists(filename):
        # Load the model if it exists
        loaded_model = joblib.load(filename)
    else:
        # Train the model if it doesn't exist
        loaded_model = train_and_save_model(model, filename)

    return loaded_model

# Load or train models
knn_model = load_or_train_model(KNN(), knn_model_filename)
decision_tree_model = load_or_train_model(DecisionTree(), decision_tree_model_filename)
naive_bayes_model = load_or_train_model(NaiveBayes(), naive_bayes_model_filename)
svm_model = load_or_train_model(SVM(), svm_model_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_fraud_knn', methods=['POST'])
def detect_fraud_knn():
    selected_model = request.form.get('model')

    if selected_model == 'KNN':
        trained_model = knn_model
    elif selected_model == 'Decision Tree':
        trained_model = decision_tree_model
    elif selected_model == 'Naive Bayes':
        trained_model = naive_bayes_model
    elif selected_model == 'SVM':
        trained_model = svm_model

    # Extracting values from the form
    data_str = request.form.get('data')

# Tìm tất cả các số thực trong chuỗi
    pattern = r"-?\d*\.?\d+"
    matches = re.findall(pattern, data_str)

# Chuyển đổi danh sách các chuỗi thành mảng numpy
    X_test = np.array(list(map(float, matches)), dtype=np.float64)

# Tạo tên cột
    column_names = ['Time'] + ['V' + str(i) for i in range(1, 29)] + ['Amount']

# Chuyển mảng thành DataFrame với 1 dòng và 30 cột
    X_test = pd.DataFrame(X_test.reshape(1, -1), columns=column_names)

    X_test['Vamount'] = scaler1.transform(X_test["Amount"].values.reshape(-1,1))

    X_test = X_test.drop(["Amount"], axis = 1)
    
    X_test = pca.transform(X_test)
    X_test = scaler2.transform(X_test)

    prediction = trained_model.predict(X_test)

    # Save prediction history
    save_prediction_history(data_str,selected_model,prediction)

    return jsonify({'result': 'Gian lận' if prediction == 1 else 'Hợp pháp'})

def save_prediction_history(features_str, selected_model, result):
    with app.app_context():  # Bao quanh hoạt động liên quan đến cơ sở dữ liệu
        new_prediction = PredictionHistory(features = features_str,model = selected_model, result='Gian lận' if result == 1 else 'Hợp pháp')
        db.session.add(new_prediction)
        db.session.commit()

@app.route('/prediction_history')
def prediction_history():
    with app.app_context():  # Bao quanh hoạt động liên quan đến cơ sở dữ liệu
        predictions = PredictionHistory.query.all()
    return render_template('history.html', predictions=predictions)



@app.template_filter('to_local_time')
def to_local_time(value):
    if value is None:
        return ""
    if isinstance(value, str):
        value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    vn_tz = timezone('Asia/Ho_Chi_Minh')  # Múi giờ Việt Nam
    value = value.replace(tzinfo=timezone('UTC')).astimezone(vn_tz)
    return value.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    with app.app_context():  # Bao quanh tạo bảng cơ sở dữ liệu
        db.create_all()
    app.run(debug=True)
