import sys, os
# sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import lime
from lime import lime_text
from sklearn.pipeline import make_pipeline

from nnlocallinear import NLS, LLS, NNPredict

os.makedirs("data", exist_ok=True)
path_ = './data/rt-polaritydata/'
def load_polarity(path=path_):
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), 'rb'):
            data.append(line.decode('utf8', errors='ignore').strip())
            labels.append(l)
    return data, labels

x, y = load_polarity()
x_train_all, x_test, y_train_all, y_test = train_test_split(
    x, y, test_size=.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_all, y_train_all, test_size=.1, random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

class VectorizeText():
        def __init__(self):
            self.count_vect = CountVectorizer()
            self.tf_transformer = TfidfTransformer(use_idf=False)
        def fit(self, x):
            x = self.count_vect.fit_transform(x)
            self.tf_transformer.fit(x)
        def transform(self, x):
            x = self.count_vect.transform(x)
            x = self.tf_transformer.transform(x)
            return x.toarray()     
        def get_feature_names(self):
            return self.count_vect.get_feature_names()

vectorizer = VectorizeText()
vectorizer.fit(x_train)    
x_vec_train = vectorizer.transform(x_train)
x_vec_test = vectorizer.transform(x_test)

names_features = np.array(vectorizer.get_feature_names())


comb_parameters = [{
        'es_give_up_after_nepochs': 20
        , 'hidden_size': 100
        , 'num_layers': 4
        , 'n_classification_labels': 2
        , 'penalization_thetas': 0.5
    }
        ]

for parameter in comb_parameters:
    model = NLS(
        verbose=0
        , es=True
        , gpu=True
        , scale_data=False
        , varying_theta0=False
        , fixed_theta0=True
        , dataloader_workers=0
        # , with_mean=False
        , **parameter
    ) 
    model.fit(x_train=x_vec_train, y_train=y_train)

y_pred = model.predict(x_vec_test)
print('F1 Score:', metrics.f1_score(y_test, y_pred, average='binary'))
print('Accuracy', metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))

def predict(texts):
    return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
    return model.predict_proba(vectorizer.transform(texts))

y_p = predict(x_test)
print('Val accuracy', metrics.accuracy_score(y_test, y_p))

x_explain = [x_test[1]]
x_vec_explain = [x_vec_test[1]]
print('x_explain:', x_explain)
print(x_explain[0])
print('Predicted class:', model.predict(x_vec_explain)[0])
print('Predict probabilities:', model.predict_proba(x_vec_explain))
print('True class:', y_test[1])

feature_names=[f'{e}: {word}' for e, word in enumerate(x_explain[0].split())]
words=np.array([f'{word}' for e, word in enumerate(x_explain[0].split())])

def get_explanation(x_vec_explain, document, num_features=10):
        
        explanation = model.get_thetas(x_pred=x_vec_explain, net_scale=True)
        betas = explanation[2][0]
        print('len(betas)', len(betas))
        words_from_text_indices = np.argwhere(x_vec_explain[0] != 0).reshape(-1)
        print('len(words_from_text_indices)', len(words_from_text_indices))
        # Prediction from the model
        prediction = model.predict(x_vec_explain).reshape(-1)
        predict_proba = model.predict_proba(x_vec_explain).reshape(-1)
        ind_pred_proba = np.argsort(predict_proba)[::-1]
        print('probabilities: ', ind_pred_proba)
        
        # col_betas = int(prediction)
        col_betas = ind_pred_proba[0]
        col_betas_neg = ind_pred_proba[1]

        betas_document = betas[words_from_text_indices, col_betas]
        print('betas_document len:', len(betas_document))
        betas_document_neg = betas[words_from_text_indices, col_betas_neg]

        betas_final = betas_document - betas_document_neg
        words_features_document = names_features[words_from_text_indices].reshape(-1)
        
        return dict(
            chi_names=words_features_document,
            chi_values=words_features_document,
            x_values=words_features_document,
            x_names=words_features_document,
            y_p=predict_prob(x_explain)[0, 1],
            y_p_max=1.0,
            y_p_min=0.0,
            y_p_local_model=predict_prob(x_explain)[0, 1],
            y_p_local_model_max=None,
            y_p_local_model_min=None,
            error=None,
            importances=betas_final,
            importances_rank1=betas_document,
            importances_rank2=betas_document_neg, 
            diff_convergence_importances=None,
            ind_class_sorted=0,
            class_names= ["probability"]
        )

explain_dict = get_explanation(x_vec_explain, x_explain, num_features=13)

from explainers.visualizations.plot_importance import ExplainGraph

fig, ax = plt.subplots(figsize=(8, 9))
ax.set_title('Importance', fontsize=25)
names = explain_dict['chi_names']
importances = explain_dict['importances']
ax = ExplainGraph.plot_feature_importance(
ax=ax, names=names, vals=importances, size_title=15)
plt.savefig('text_explanation.pdf', dpi=300)