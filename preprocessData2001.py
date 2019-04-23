import os
import re
from bs4 import BeautifulSoup
import string

def cleansummary(text):
    x = re.compile('<.*?>')
    if re.search(x, text):
        return ''
    text = text.replace('\n',' ').replace('\t','')
    text = text.lstrip()
    text = text.replace('``',"\"")
    text = text.replace("''", '\"')
    re.sub( '\s+', ' ', text ).strip() #remove extra spaces
    text = text.replace()
    text = text.lower()
    return text

def make_txt_files(dirpath, doc_folderpath, sum_folderpath):
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dirpath) for f in filenames]
    doc_list = [file for file in file_list if '/docs' in file]
    sum_list = [file for file in file_list if 'perdoc' in file]
    
    for doc in doc_list:
        file = open(doc,'r')
        bs = BeautifulSoup(file,'lxml')
        file_out = open(doc_folderpath+'/'+os.path.basename(doc)+'.txt','w')
        
        for text in bs.find_all('text'):
            text = cleansummary(text.text)
            file_out.write(text)
        
        print('created: '+ doc_folderpath+'/'+os.path.basename(doc)+'.txt')
        file_out.close()
    

    for summs in sum_list:
        file = open(summs, 'r')
        bs = BeautifulSoup(file, 'lxml')
        for text in bs.find_all('sum'):
            attrs = text.attrs 
            file = attrs['docref']
            file_out = open(sum_folderpath+'/'+file+'.txt','w')
            text= cleansummary(text.text)
            file_out.write(text)
            file_out.close()

def make_test_files(dirpath, doc_folderpath, sum_folderpath):
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dirpath) for f in filenames]
    file_list = [file for file in file_list if not '.tar.gz' in file]
    doc_list = [file for file in file_list if '/docs' in file]
    sum_list1 = [file for file in file_list if 'original.summ' in file]
    sum_list = [file for file in sum_list1 if 'perdoc' in file]
     
    
    for doc in doc_list:
        file = open(doc,'r')
        bs = BeautifulSoup(file,'lxml')
        file_out = open(doc_folderpath+'/'+os.path.basename(doc)+'.txt','w')
        
        for text in bs.find_all('text'):
            text = cleansummary(text.text)
            file_out.write(text)
        
        print('created: '+ doc_folderpath+'/'+os.path.basename(doc)+'.txt')
        file_out.close()
    

    for summs in sum_list:
        file = open(summs, 'r')
        bs = BeautifulSoup(file, 'lxml')
        for text in bs.find_all('sum'):
            attrs = text.attrs 
            file = attrs['docref']
            file_out = open(sum_folderpath+'/'+file+'.txt','w')
            text= cleansummary(text.text)
            file_out.write(text)
            file_out.close()
            
            
def split_new(output_path, split):
    training = './training'
    test = './test'
    
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(training) for f in filenames]
    doc_list = [file for file in file_list if '_docs' in file]
    sum_list = [file for file in file_list if '_sums' in file]
    
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(test) for f in filenames]
    doc_list.extend([file for file in file_list if '_docs' in file])
    sum_list.extend([file for file in file_list if '_sums' in file])
    
    total_files = set()
    for i in range(len(sum_list)):
        for filename in doc_list:
            if os.path.basename(filename) == '.DS_Store':
                continue
            elif os.path.basename(sum_list[i]) in filename:
                total_files.add(os.path.basename(filename))
    print(len(x))
    return total_files


    
#####

training_path = './DUC/DUC2001_Summarization_Documents/data/training'

#output_path_training_docs = './training/2001_docs'
#output_path_training_summ = './training/2001_sums'
#make_txt_files(training_path, output_path_training_docs, output_path_training_summ)

testing_path = './DUC/DUC2001_Summarization_Documents/data/test'
#output_path_testing_docs = './test/2001_docs'
#output_path_testing_summ = './test/2001_sums'
#make_test_files(testing_path, output_path_testing_docs, output_path_testing_summ)

new_output = './2001_new_split'
split = .7
x = split_new(new_output,split)