from time import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import fbeta_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

from IPython.display import display

class LearningModel:
    """ ........... """
    features = []
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    training_time = None

    predictions = {}
    y_pred = None
    y_pred_proba = None
    y_actual = None

    evaluation = {}
    eval_report = None
    feature_scores = None

    valid_train_set = False
    valid_test_set = False
    valid_predictions = False

    estimator = None

    def __init__(self, name, learner, model_type='classification', verbose=False):
        '''
        inputs:
           - name: model name
           - learner: learning algorithm or pipeline
        '''
        self.name = name
        self.learner = learner
        self.learner_class = learner.__class__.__name__
        self.model_type = model_type
        self.verbose = verbose

    def fit_and_predict(self, X_train, X_test, y_train, y_test, evaluate=True):
        self.add_data(X_train, X_test, y_train, y_test)
        self.fit_model()
        self.make_predictions()
        if evaluate:
            self.evaluate_model()

    def fit_model(self):
        if not self.valid_train_set:
            return
        if self.verbose:
            print('Fitting Model')
        learner = self.learner

        # Fit the learner to the training set of 'sample_size'
        start = time() # Get start time
        learner.fit(self.X_train, self.y_train)
        end = time() # Get end time

        # Calculate the time it took to fit the model on training data
        self.training_time = round(end - start)

        if learner.__class__.__name__ == 'Pipeline':
            pipeline_learner = 'clf'
            self.estimator = learner.get_params()[pipeline_learner]
        elif learner.__class__.__name__ == 'GridSearchCV':
            self.estimator = learner.best_estimator_
        else:
            self.estimator = learner

    def make_predictions(self):
        if self.valid_test_set:
            if self.verbose:
                print('Making Predictions')
            self.y_pred = self.learner.predict(self.X_test)
            self.y_pred_proba = self.learner.predict_proba(self.X_test)
            self.valid_predictions = True

    def basic_clf_eval(self, y_pred, y_pred_proba, y_actual):
        model_eval = pd.Series()

        # Calculate the time it took to fit the model on training data
        model_eval['FitTime'] = self.training_time

        # Compute metrics on test set predictions
        model_eval['Accuracy'] = accuracy_score(y_pred=y_pred, y_true=y_actual, normalize=True)
        model_eval['FBeta'] = fbeta_score(y_pred=y_pred, y_true=y_actual, beta=2, pos_label=1)
        model_eval['F1'] = f1_score(y_pred=y_pred, y_true=y_actual, pos_label=1)
        model_eval['AUC'] = roc_auc_score(y_score=y_pred_proba, y_true=y_actual, average='weighted')

        # Make DataFrame object from metrics
        model_eval_df = pd.DataFrame(model_eval, columns=[self.name]).T

        return model_eval_df

    def evaluate_classifier(self, y_pred, y_pred_proba, y_actual):
        eval_metrics = self.basic_clf_eval(y_pred, y_pred_proba, y_actual)
        self.evaluation['metrics'] = eval_metrics
        self.eval_report = eval_metrics
        self.evaluation['classification_report'] = classification_report(y_actual, y_pred)

    def plot_roc_curve(self, pred_proba, pos_class=1):
        pred_proba = self.y_pred_proba
        y_actual = self.y_actual
        if isinstance(pred_proba[0], np.ndarray):
            scores = pred_proba[:, pos_class]
        else:
            scores = pred_proba

        roc_auc = roc_auc_score(y_actual, scores)
        fpr, tpr, thresholds = roc_curve(y_true=y_actual, y_score=scores, pos_label=pos_class)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def add_feature_scores(self, scores):
        if len(scores) == len(self.features):
            self.feature_scores = (
                pd.DataFrame(data=list(zip(self.features, scores)),
                             columns=['feature', 'score'])
                .sort_values('score', ascending=False)
                .reset_index(drop=True))

    def display_top_features(self, top_n=10):
        if self.feature_scores is not None:
            scores = self.feature_scores.copy()
            scores.columns = ['Feature', 'Score']
            scores = scores.set_index(
                np.array([str(i) for i in range(1, len(scores)+1)]))
            display(scores[:top_n])

    def plot_top_features(self, top_n=5):
        if self.feature_scores is not None:
            disp_title = 'Top ' + str(top_n) + ' Model Features ' + self.name
            _ = sns.barplot(x=self.feature_scores.feature[:top_n],
                            y=self.feature_scores.score[:top_n])
            for item in _.get_xticklabels():
                item.set_rotation(45)
            _.set(xlabel='Feature', ylabel='Mean Score', title=disp_title)
            plt.show()

    def evaluate_model(self, eval_fun=None, evaluation_set='test'):
        scores = getattr(self.estimator , "feature_importances_", None)
        if self.verbose:
            print('Evaluating Model')
        if scores is not None:
            self.add_feature_scores(scores)
        if self.valid_predictions:
            y_actual = self.y_actual
            y_pred = self.y_pred
            y_pred_proba = self.y_pred_proba[:, 1]
            self.evaluate_classifier(y_pred, y_pred_proba, y_actual)

    def display_evaluation(self, plot_roc=True):
        display(self.evaluation['metrics'])
        print(self.evaluation['classification_report'])
        if plot_roc:
            self.plot_roc_curve(self.y_pred_proba)

    def add_data(self, X_train, X_test, y_train, y_test, infer_features=True):
        '''
        inputs:
           - X_train: features (training set)
           - y_train: target (training set)
           - X_test: features (test set)
           - y_test: target (test set)
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_actual = y_test

        if infer_features:
            self.infer_features()

        self.validate_train_data()
        self.validate_test_data()

    def infer_features(self):
        if self.X_train.__class__.__name__ == 'DataFrame':
            self.features = self.X_train.columns.tolist()
        elif self.X_test.__class__.__name__ == 'DataFrame':
            self.features = self.X_test.columns.tolist()

    def validate_test_data(self):
        test_set = [self.X_test, self.y_test]
        self.valid_test_set = all([x is not None for x in test_set])

    def validate_train_data(self):
        train_set = [self.X_train, self.y_train]
        self.valid_train_set = all([x is not None for x in train_set])


def eval_db(current_db, to_add=None, display_db=False, force_add=False, reset_db=False):
    if to_add is not None:
        if len(current_db.columns) == len(to_add.columns):
            if to_add.index.isin(current_db.index).any():
                if force_add:
                    current_db = pd.concat([current_db, to_add])
                else:
                    idx_in = to_add.index[to_add.index.isin(model_evals.index)]
                    current_db.loc[idx_in] = to_add.loc[idx_in]
            else:
                current_db = pd.concat([current_db, to_add])
        else:
            print('Error: Length is not the same')
            return current_db
    if reset_db:
        current_db = to_add
    if display_db:
        display(current_db)
    return current_db
