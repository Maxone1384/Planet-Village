from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_models(X, y):
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_score = 0

    # ----- مدل SVM -----
    print("Training SVM...")
    svm_params = {'C':[1,10], 'gamma':[0.01, 0.001], 'kernel':['rbf']}
    svm = GridSearchCV(SVC(), svm_params, cv=3, n_jobs=-1)
    svm.fit(X_train, y_train)
    print("SVM best score:", svm.best_score_)

    if svm.best_score_ > best_score:
        best_score = svm.best_score_
        best_model = svm.best_estimator_

    # ----- مدل Random Forest -----
    print("Training Random Forest...")
    rf_params = {'n_estimators':[100,200], 'max_depth':[None,20]}
    rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=3, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("RF best score:", rf.best_score_)

    if rf.best_score_ > best_score:
        best_score = rf.best_score_
        best_model = rf.best_estimator_

    # ----- مدل KNN -----
    print("Training KNN...")
    knn_params = {'n_neighbors':[3,5,7]}
    knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, n_jobs=-1)
    knn.fit(X_train, y_train)
    print("KNN best score:", knn.best_score_)

    if knn.best_score_ > best_score:
        best_score = knn.best_score_
        best_model = knn.best_estimator_

    # ذخیره بهترین مدل
    with open('model/final_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print("Best model saved. Score:", best_score)
    return best_model, X_test, y_test
