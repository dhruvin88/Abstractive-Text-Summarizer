from utils import *
import tensorflow as tf
from pyrouge import Rouge155

#load test articles and summaries
test_sums = load_pickle('./data/cnn_test_sums.p')
test_texts = load_pickle('./data/cnn_test_articles.p')

word_embedding_matrix = load_pickle('./data/cnn_embedding_martix.p')
vocab_to_int = load_pickle('./data/cnn_vocab_to_int.p')
int_to_vocab = load_pickle('./data/cnn_int_to_vocab.p')

int_summaries, _, _ = convert_to_ints(test_sums, 0, 0, vocab_to_int)
int_texts, _, _ = convert_to_ints(test_texts, 0, 0, vocab_to_int,eos=True)

gen_sum_length = [100] * len(int_texts)

checkpoint = "./checkpoint_model"

if type(gen_sum_length) is list:
    if len(int_texts)!=len(gen_sum_length):
        raise Exception("[Error] make Summaries parameter gen_sum_length must be same length as input_sentences or an integer")
    gen_summary_length_list = gen_sum_length
else:
    gen_summary_length_list = [gen_sum_length] * len(int_texts)

loaded_graph = tf.Graph()
batch_size = 10

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    output_file = open('output.txt','w',encoding='utf-8')
    for i, text in enumerate(int_texts):
        generagte_summary_length = gen_summary_length_list[i]
        answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                          summary_length: [generagte_summary_length],
                                          text_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0]
        
    # Remove the padding from the summaries
        pad = vocab_to_int["<PAD>"]

        #Save Gold Summary
        gold_path = './Gold_Summaries/'
        # Create target Directory if don't exist
        if not os.path.exists(gold_path):
            os.mkdir(gold_path)
            print("Directory " , gold_path ,  " Created ")
        else:    
            print("Directory " , gold_path ,  " already exists")

        gold_out = open(gold_path+'Gold.A.'+str(i+1)+'.txt', 'w', encoding='utf-8')
        for word in test_sums[i]:
            if word == '.':
                word ='.\n'
                gold_out.write(word)
            else: 
                gold_out.write(word+" ")
        gold_out.close()
        
        output_file.write("- Article:\n")
        for word in test_texts[i]:
            if word == '.':
                word ='.\n'
                output_file.write(word)
            else: 
                output_file.write(word+" ")
        
        output_summary = " ".join([int_to_vocab[i] for i in answer_logits if i != pad])

        #Save Generate summary
        Gen_path = './Model_Summaries/'
        # Create target Directory if don't exist
        if not os.path.exists(Gen_path):
            os.mkdir(Gen_path)
            print("Directory " , Gen_path ,  " Created ")
        else:    
            print("Directory " , Gen_path ,  " already exists")

        gen_out = open(Gen_path+"Model."+str(i+1)+'.txt', 'w', encoding='utf-8')
        for word in answer_logits:
            word = int_to_vocab[word]
            if word == '.':
                word ='.\n'
                gen_out.write(word)
            elif word == '<PAD>':
                continue
            else: 
                gen_out.write(word+" ")
        gen_out.close()

        output_file.write('\n- Summary: \n' + output_summary+'\n\n\n\n')

        print('- Summary:\n\r {}\n\r\n\r'.format(output_summary))

    output_file.close()
