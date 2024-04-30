from classifier import Classifier,Methods
import pickle
import random

def main():
    full_features = pickle.load(open("D:\Conex\PreSage\Prototypes\Dataset_10h_wk1\images\\eval.pickle","rb"))
    features = []
    labels = []
    for id in full_features.keys():
        features.append(full_features[id]["feat"])
        labels.append(full_features[id]["label"])
    clf = Classifier()
    clf.set(method=Methods.MEAN)
    clf.model_file = "model_svm.mean"
    train_ids = list(range(0,len(features)))
    random.shuffle(train_ids)
    train_features = []
    train_labels = []
    eval_features = []
    eval_labels = []
    n = 0
    for id in train_ids:
        if len(train_features) <= int(len(features)/2):
            train_features.append(features[id])
            train_labels.append(labels[id])
        else:
            eval_features.append(features[id])
            eval_labels.append(labels[id])
    clf.train(train_features, train_labels)
    accuracy = clf.evaluate(eval_features, eval_labels)
    print("accuracy ",accuracy)

if __name__ == '__main__':
    main()