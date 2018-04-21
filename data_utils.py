from gensim.parsing import PorterStemmer
from nltk import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors

def words_to_glove(sentences, model):
    filename = 'glove.6B.100d.txt'  # add [voc_size dimension_size]. 400000 50/100/200/300 to the top file,
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    words = []
    sents = []
    for sentence in sentences:
        for word in sentence:
            try:
                word_vector =model[word]
                #print(word_vector)
                #word_vector = model['a']
            except:
                print('Found unknown word: {0}'.format(word))
                word_vector = model['a']
                #print(word_vector)
            #words.append(word_vector)
        #sents.append((words))
            sents.append((word_vector)) #
        words = []
    return sents

def words_to_word2vec(sentences, model):
    words = []
    sents = []
    for sentence in sentences:
        for word in sentence:
            try:
                word_vector =model[word]
                #print(word_vector)
                #word_vector = model['a']
            except:
                print('Found unknown word: {0}'.format(word))
                word_vector = model['a']
                #print(word_vector)
            #words.append(word_vector)
        #sents.append((words))
            sents.append((word_vector)) #
        words = []
    return sents


def sentences_to_lists(filename):
    filename = open(filename, 'r')
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in filename:
        if (('-DOCSTART-') in line):
            continue
        tokens = line.split(' ')
        #print(tokens[0])
        if len(tokens) > 3:
            sentence.append(tokens[0])
            label.append(tokens[3].replace('\n', ''))
        if len(tokens)  < 3 and len(sentence) != 0:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
    return  sentences, labels

def labels_to_integer(labels):
    tags = []
    tag = []
    for label in labels:
        for word in label:
            if word == 'PER':
                #tag.append(0.0)
                tags.append(0.0)#
            elif word == 'ORG':
                #tag.append(1.0)
                tags.append(1.0)#
            elif word == 'LOC':
                #tag.append(2.0)
                tags.append(2.0)#
            elif word == 'MISC':
                #tag.append(3.0)
                tags.append(3.0)#
            elif word == 'O':
                #tag.append(8.0)
                tags.append(4.0)#
            else:
                print('Catching different tag: {0}'.format(word))
        #tags.append(tag)
        tag = []
    return tags

def integer_to_label(word):
    if word == 0.0:
        #tag.append(0.0)
        return "PER"
    elif word == 1.0:
        #tag.append(1.0)
        return 'ORG'
    elif word == 2.0:
        #tag.append(2.0)
        return 'LOC'
    elif word == 3.0:
        #tag.append(3.0)
        return 'MISC'
    elif word == 4.0:
        #tag.append(8.0)
        return 'O'
    else:
        print('Catching different tag: {0}'.format(word))
        #tags.append(tag)
        return 'Error'
