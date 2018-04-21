from itertools import chain
import nltk
import eli5
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import eli5
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

def get_sentences_and_NER(filename):
    sentence = []
    sentences = []
    for line in filename:
        if (('-DOCSTART-') in line):
            continue
        line = line.split(' ')
        #print(len(line))

        if(len(line) > 1):
            tuple = (line[0], line[2], line[3].replace('\n',''))
            sentence.append(tuple)
        if(len(line) == 1):
            sentences.append(sentence)
            sentence = []
    return sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


file_train = open('datasets/conll2003/eng.train', 'r')
train_sents = get_sentences_and_NER(file_train)



file_test = open('datasets/conll2003/eng.testa', 'r')
test_sents = get_sentences_and_NER(file_test)
#print(test_sents)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

print(y_train[0])
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
#print(X_test)
print(X_test[1])


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)

labels = list(crf.classes_)
#labels.remove('O')

print(labels)

y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))
import sys
sys.exit()
print(eli5.format_as_text((eli5.explain_weights(crf, top=30))))

'''''
eli5.show_weights(crf, top=5, show=['transition_features'])
eli5.show_weights(crf, top=10, targets=['O', 'B-ORG', 'I-ORG'])
eli5.show_weights(crf, top=10, feature_re='^word\.is',
                  horizontal_layout=False, show=['targets'])
expl = eli5.explain_weights(crf, top=5, targets=['O', 'B-LOC', 'I-LOC'])
print(eli5.format_as_text(expl))
'''''