from classifier import Classifier,Methods
import pickle
import random

def main():
    full_features = pickle.load(open("D:\Conex\PreSage\Prototypes\Dataset_10h_wk1\images\\train.pickle","rb"))
    features = []
    labels = []
    exclude_objects = ["bed"]
    positive = ["person","person_eating","person_sitting","person_sleeping","person_standing"]
    negative = ["food","food_cold","food_hot"]
    positive_lbl = "person"
    negative_lbl = "non-person"
    for id in full_features.keys():

        lbl = full_features[id]["label"]
        if lbl in positive:
            lbl = positive_lbl
        elif lbl in negative:
            lbl = negative_lbl
        else:
            continue

        features.append(full_features[id]["feat"])
        labels.append(lbl)

    clf = Classifier()
    clf.set(method=Methods.MEAN)
    clf.model_file = "../output/model_mean.sav"
    train_ids = list(range(0,len(features)))
    random.shuffle(train_ids)
    train_features = []
    train_labels = []
    eval_features = []
    eval_labels = []
    n = 0
    for id in train_ids:
        if len(train_features) <= 2*int(len(features)/3):
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