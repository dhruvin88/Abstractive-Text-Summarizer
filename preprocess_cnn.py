import os
import re
from utils import *
import tqdm
import math

def load_doc(filename):
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text

def split_article(doc):
    index = doc.find('@highlight')
    article, summary = doc[:index], doc[index:].split('@highlight')
    summary = " ".join([s.strip() for s in summary if len(s) > 0])
    #article = " ".join([a.strip() for a in article if len(a) > 0])
    return article, summary

def clean(text):
    text = text.replace('\n',' ').replace('\t','')
    text = text.lstrip()
    text = text.replace('``',"\"")
    text = text.replace('``','\"')
    text = text.replace("''", '\"')
    text = text.replace('(CNN) -- ','')
    text = text.replace('(CNN)','')
    re.sub( '\s+', ' ', text ).strip() #remove extra spaces
    text = text.lower()
    return text

def load_articles(path):
    articles = list()
    summaries = list()
    files = listdir(path)
    for i in tqdm.tqdm(range(len(files))):
        filename = path + '/'+files[i]
        doc = load_doc(filename)
        article, summary = split_article(doc)
        article = clean(article)
        summary = clean(summary)
        if article and summary:
            articles.append(get_tokens(article))
            summaries.append(get_tokens(summary))
    return articles, summaries

path = '/Users/Dhruvin/Desktop/cnn/stories'
articles, summaries = load_articles(path)
print('Loaded Stories %d' % len(articles))

new_total = math.ceil(len(articles)*.15)

train_num = math.ceil(new_total*.7)

train_articles = articles[:train_num]
test_articles = articles[train_num:new_total]

train_sums = summaries[:train_num]
test_sums = summaries[train_num:new_total]

# Create target Directory if don't exist
if not os.path.exists('./data'):
    os.mkdir('./data')
    print("Directory " , './data' ,  " Created ")
else:    
    print("Directory " , './data' ,  " already exists")

save_pickle("./data/cnn_train_articles.p",train_articles)
save_pickle("./data/cnn_test_articles.p",test_articles)

save_pickle("./data/cnn_train_sums.p",train_sums)
save_pickle("./data/cnn_test_sums.p",test_sums)

#Count word frequency in articles and summaries
word_counts = {}
word_counts = count_words(word_counts, train_articles)
word_counts = count_words(word_counts, train_sums)
print("Size of Vocabulary:", len(word_counts))

#load GloVe
embd = loadGloVe('glove.6B.50d.txt')

## find unknown words
unk_words = []
unk_list = open('unknown_list2.txt', 'w', encoding='utf-8')
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

int_summaries, word_count, unk_count = convert_to_ints(train_sums, word_count, unk_count, vocab_to_int)
int_texts, word_count, unk_count = convert_to_ints(train_articles, word_count, unk_count, vocab_to_int,eos=True)

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

save_pickle("./data/cnn_train_int_summaries.p",int_summaries)
save_pickle("./data/cnn_train_int_texts.p",int_texts)
save_pickle("./data/cnn_word_embedding_matrix.p",word_embedding_matrix)

save_pickle("./data/cnn_vocab_to_int.p",vocab_to_int)
save_pickle("./data/cnn_int_to_vocab.p",int_to_vocab)