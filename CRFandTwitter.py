from itertools import chain
import nltk
import eli5
import re
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import eli5
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import sys
import warnings


warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
myre = re.compile("["
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  "]+", flags=re.UNICODE)

def clean_str(s):
    s = myre.sub('', s)
    s = re.sub('["\()?$]', '', s)
    return s

def before_word(tokens):
    if len(tokens) > 0:
        words = ' '.join(tokens.split())
        words = words.split(' ')
        if(len(words) > 0):
            return words[-1]
    return ''
def after_word(tokens):
    if len(tokens) > 0:
        words = ' '.join(tokens.split())
        words = words.split(' ')
        if (len(words) > 0):
            return words[0]
    return ''

def get_sentences_and_twitter(filename):
    sentence = []
    i = 0
    sentences = []
    for ind, line in enumerate(filename):
        tokens = re.split("\|(.*?)\|", line)
        #print(len(line))
        print(tokens)
        tokens[2]= re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tokens[2])  # sentence
        #tokens[2] = re.sub(r'\bit+:\[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tokens[2])  # sentence
        #tokens[2] = re.sub(r'\t.co+:\[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tokens[2])  # sentence
        tokens[2] = clean_str(tokens[2])
        print(tokens[2])
        words = tokens[2].split(tokens[3])
        print(words)
        print('before: {0}, word: {1}, after: {2}'.format(before_word(words[0]), tokens[3], after_word(words[1])))
        print(tokens[3])  # word
        print(tokens[4].replace('|\n',''))  # NER
        print('*******')
        print(ind)
        if (ind == 2100):
            sys.exit()

    sys.exit()
    return sentences #536640503643918000, 538084379046969000

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
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


file_train = open('datasets/neel/twitter_test.txt', 'r', encoding='utf8')
train_sents = get_sentences_and_twitter(file_train)
print(len(train_sents))
#print(train_sents)
#print(train_sents[0])


sys.exit()


file_test = open('datasets/neel/twitter_test.txt', 'r', encoding='utf8')
test_sents = get_sentences_and_twitter(file_test)
print(test_sents)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

print(X_test[5])
#print(y_train)
import sys
sys.exit()
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

print(eli5.format_as_text((eli5.explain_weights(crf, top=30))))


