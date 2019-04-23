from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
import nltk

def loadGloVe(filename):
    embd = {}
    file = open(filename,'r',encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab = row[0]
        embd[vocab] = row[1:]
    print('GloVe Loaded.')
    file.close()
    return embd


def count_words(count_dict, text):
    for sentence in text:
        for word in sentence:
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
    return count_dict

def convert_to_ints(text, word_count, unk_count, vocab_to_int,eos=False,sos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        if sos:
            sentence_ints.append(vocab_to_int['<SOS>'])
        for word in sentence:
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

def save_pickle(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    print("Saved: "+filename)
    save_stuff.close()

def load_pickle(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    print("Loaded: "+filename)
    saved_stuff.close()
    return stuff

def get_docs_and_sums(path_docs, path_sums):
    #filenames of docs and summaries
    doc_names = [f for f in listdir(path_docs) if isfile(join(path_docs, f))]
    sums_names = [f for f in listdir(path_sums) if isfile(join(path_sums, f))]

    #get files in both doc and summaries
    test_files = []
    for file in sums_names:
        if file in doc_names and file != '.DS_Store':
            test_files.append(file)

    docs = []
    sums= []

    #parse docs and summaries
    for file in test_files:
        doc = open(path_docs+'/'+file,'r')
        sum_doc = open(path_sums+'/'+file, 'r')

        doc_text = doc.readlines()
        if doc_text  == []:
            continue
        else:    
            doc_text = get_tokens(doc_text[0])
            
            sum_text = sum_doc.readlines()
            sum_text = get_tokens(sum_text[0])

            docs.append(doc_text)
            sums.append(sum_text)

    doc.close()
    sum_doc.close()
    return docs, sums

def get_tokens(text):
    tokenized_text = []
    sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        tokenized_text.extend(nltk.word_tokenize(sentence))
    return tokenized_text

def get_vocab_to_int(word_counts,embd):
    #dictionary to convert words to integers
    vocab_to_int = {}
    # Index words from 0
    value = 0

    for word, count in word_counts.items():
        if word in embd:
            vocab_to_int[word] = value
            value += 1
    
    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<SOS>"]

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    return vocab_to_int

def get_int_to_vocab(vocab_to_int):
    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word
    return int_to_vocab


def get_word_embedding_matrix(vocab_to_int,embd):
    # Need to use 50 for embedding dimensions to match GloVe's vectors.
    embedding_dim = 50
    glove_words = len(vocab_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((glove_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embd:
            word_embedding_matrix[i] = embd[word]
        else:
            # If word not in GloVe, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embd[word] = new_embedding
            word_embedding_matrix[i] = new_embedding
    return word_embedding_matrix


def find_unk_words(word_counts, embd):
    words = []
    for word in word_counts.keys():
        if word not in embd:
            words.append(word)
    return words

def create_and_save_embedding_matrix(word2int,
                                     embd,
                                     save_path,
                                     embedding_dim=50):
    """creates embedding matrix for each word in word2int. if that words is in
       pretrained_embeddings, that vector is used. otherwise initialized
       randomly.
    """
    embedding_matrix = np.zeros((len(word2int), embedding_dim), dtype=np.float32)
    for word, i in word2int.items():
        if word in embd.keys():
            embedding_matrix[i] = embd[word]
        else:
            embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embedding_matrix[i] = embedding
    save_pickle("./data/cnn_embedding_martix.p",embedding_matrix)
    return np.array(embedding_matrix)


def pad_sentence_batch(sentence_batch,vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(summaries, texts, batch_size,vocab_to_int):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size

        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch,vocab_to_int))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch,vocab_to_int))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

###################
