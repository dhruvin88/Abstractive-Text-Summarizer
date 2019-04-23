import os
path = './Model_Summaries/'
for filename in os.listdir(path):
    file_start = filename[:6]
    file_end = filename[6:]
    while (len(file_end) <= 8):
        file_end = '0'+ file_end

    #print(file_start+file_end)
    os.rename(path+filename, (path+file_start+file_end))
 

path = './Gold_Summaries/'
for filename in os.listdir(path):
    file_start = filename[:7]
    file_end = filename[7:]
    while (len(file_end) <= 8):
        file_end = '0'+ file_end
        
    #print(file_start+file_end)
    os.rename(path+filename, (path+file_start+file_end))
 