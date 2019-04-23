import os
import re

def cleansummary(text):
    x = re.compile('<.*?>')
    if re.search(x, text):
        return ''
    text = text.replace('\n',' ').replace('\t','')
    text = text.lstrip()
    text = text.replace('``',"\"")
    text = text.replace("''", '\"')
    re.sub( '\s+', ' ', text ).strip() #remove extra spaces
    text = text.lower()
    return text

def make_txt_files(dirpath, folderpath,subFolders):
    file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dirpath) for f in filenames]
    for filename in file_list:
        sub_folder_name = ''
        if subFolders:
            sub_folder_name = filename[95:102]+'.'
        
        if not os.path.basename(filename) == '.DS_Store':
            file = open(filename, "r")
            print(filename)
            fileout = open(folderpath + sub_folder_name + os.path.basename(filename) + ".txt",'w')
            
            for line in file:
                fileout.write(cleansummary(line))
            file.close()
            fileout.close()
            
            print("created: " + fileout.name )
#####

dir_path1 = "./DUC/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
folder_path = './documents/'
make_txt_files(dir_path1, folder_path, True)

dir_path2 = './DUC/duc2004_results/ROUGE/eval_2/models'
folder_path = './summaries/'
make_txt_files(dir_path2, folder_path,False)

#dir_path3 = './duc2004_results/ROUGE/eval_3/peers'
#folder_path = './output_summaries/'
#make_txt_files(dir_path3, folder_path,false)
