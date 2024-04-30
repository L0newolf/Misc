from sklearn import model_selection, svm
import numpy as np
import multiprocessing
import pickle
from scipy.spatial import distance

debug = False

class Methods:
    _min = 0
    _max = 6
    MEAN,PCA,LDA,SVM,NB,NN = range(_min,_max)
    @staticmethod
    def validate(v):
        # return True
        if v in [Methods.SVM, Methods.MEAN, Methods.PCA, Methods.LDA, Methods.NB, Methods.NN]:
            return True
        else:
            return False

class DISTANCE:
    min = 0
    max = 2
    EUCLIDEAN,COSINE = range(min,max)
    @staticmethod
    def validate(v):
        if v in [DISTANCE.EUCLIDEAN, DISTANCE.COSINE]:
            return True
        else:
            return False

class Classifier(object):
    def __init__(self):
        self.use_cv = False
        self.method = Methods.MEAN
        self.distance = DISTANCE.COSINE
        self.clf = None
        self.templates = []
        self.train_method = {
            Methods.SVM : self.train_SVM,
            Methods.MEAN : self.train_MEAN
        }
        self.eval_method = {
            Methods.SVM : self.eval_SVM,
            Methods.MEAN : self.eval_MEAN
        }
        self.model_file = "model.sav"
        self.labels_dict = dict()
        self.reverse_labels = dict()

    def save(self):
        data = dict()
        data["labels"] = self.labels_dict
        data["rlabels"] = self.reverse_labels
        if self.method in [Methods.SVM]:
            data["model"] = self.clf
        else:
            data["model"] = self.templates
        pickle.dump(data,open(self.model_file,"wb"))

    def load(self):
        data = pickle.load(open(self.model_file,"rb"))
        self.labels_dict = data["labels"]
        self.reverse_labels = data["rlabels"]
        if self.method in [Methods.SVM]:
            self.clf = data["model"]
        else:
            self.templates = data["model"]

    def get_distance(self, X,Y):
        if self.distance == DISTANCE.COSINE:
            d = 'cosine'
        elif self.distance == DISTANCE.EUCLIDEAN:
            d = 'euclidean'
        return distance.cdist(X,Y,d).diagonal()

    def svc_param_selection(self, X, y, nfolds=5):
        param_grid = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.1, 1],
                      'C': [0.01, 0.1, 1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}
        cores = multiprocessing.cpu_count()
        print("CPU CORES: %d" %cores)
        gridsearch = model_selection.GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=cores, cv=nfolds)  # ,verbose = 5)
        gridsearch.fit(X, y)
        gridsearch.best_params_
        return gridsearch.best_params_

    def train_SVM(self, features, ids):
        param = self.svc_param_selection(features, ids)
        print("Selected parameters " + param)
        if param["kernel"] == "rbf":
            self.clf = svm.SVC(probability=True, C=param["C"], kernel=param["kernel"],
                          gamma=param["gamma"])
        else:
            self.clf = svm.SVC(probability=True, C=param["C"], kernel=param["kernel"])
        self.clf.fit(features, ids)
        return self.clf

    def eval_SVM(self,features,ids):
        pred = self.clf.predict(features)
        return self.accuracy(pred,ids)

    def train_MEAN(self, features, ids):
        y = []
        for i in range(0, features.shape[0]):
            if ids[i] not in self.labels_dict.keys():
                self.labels_dict[ids[i]] = len(y)
                self.reverse_labels[len(y)] = ids[i]
                y.append([])
            y[self.labels_dict[ids[i]]].append(features[i])
        self.templates = []
        y = np.array(y)
        for key in range(0,len(y)):
            t = np.mean(np.array(y[key]),0)
            self.templates.append(t)
        self.templates = np.array(self.templates)
        if debug == True:
            print("Classes", self.labels_dict.keys())
            print("Templates shape: ", self.templates.shape)
        return self.templates

    def predict_MEAN(self, features):
        [r, c] = self.templates.shape
        pred = []
        if debug == True:
            print("Num samples", len(features))
        for f in features:
            A = self.get_distance(self.templates,np.tile(f, (r, 1)))
            pred.append(self.reverse_labels[A.argmin()])
        return pred

    def eval_MEAN(self,features,ids):
        pred = self.predict_MEAN(features)
        return self.accuracy(pred,ids)

    def accuracy(self, pred, actual):
        res = np.zeros(len(actual))
        for i in range(0, len(actual)):
            print(pred[i], actual[i])
        res = [1 for i in range(0,len(actual)) if pred[i]==actual[i]]
        return np.mean(res)

    def train(self, features, labels):
        features = np.array(features)
        labels = np.array(labels)
        self.train_method[self.method](features,labels)
        self.save()

    def evaluate(self,features,labels):
        return self.eval_method[self.method](features,labels)

    def set(self, dist = None, method = None):
        if (not method == None) and Methods.validate(method):
            self.method = method
        if (not dist == None) and DISTANCE.validate(dist):
            self.distance = dist

