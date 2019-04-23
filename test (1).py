from os import listdir
from utils import *
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle

###############################################################################
#path to docs and summaries
path_docs = './training/2001_docs'
path_sums = './training/2001_sums'
docs, sums = get_docs_and_sums(path_docs, path_sums)

#Count word frequency in articles and summaries
word_counts = {}
word_counts = count_words(word_counts, docs)
word_counts = count_words(word_counts, sums)
print("Size of Vocabulary:", len(word_counts))

#load GloVe
embd = loadGloVe('glove.6B.50d.txt')


## find unknown words
unk_words = []
unk_list = open('unknown_list.txt', 'w')
for doc in docs:
    unk_words.extend(find_unk_words(word_counts,embd))
for word in unk_words:
    unk_list.write(word+"\n")
unk_list.close()


vocab_to_int = get_vocab_to_int(word_counts,embd)
int_to_vocab = get_int_to_vocab(vocab_to_int)

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


word_embedding_matrix = get_word_embedding_matrix(vocab_to_int, embd)
print(len(word_embedding_matrix))

####
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(sums, word_count, unk_count,vocab_to_int)
int_texts, word_count, unk_count = convert_to_ints(docs, word_count, unk_count, vocab_to_int, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

#get stat about sum and text length
lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())

max_doc_length = lengths_texts.max
max_sum_length = lengths_summaries.max

matrix_embedding=create_and_save_embedding_matrix(vocab_to_int,embd,'./data/')

save_pickle("./data/sums.p",sums)
save_pickle("./data/docs.p",docs)

save_pickle("./data/int_summaries.p",int_summaries)
save_pickle("./data/int_texts.p",int_texts)
save_pickle("./data/word_embedding_matrix.p",word_embedding_matrix)

save_pickle("./data/vocab_to_int.p",vocab_to_int)
save_pickle("./data/int_to_vocab.p",int_to_vocab)
