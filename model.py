import numpy as np
import tensorflow as tf
import time
from tensorflow.python.layers.core import Dense
from utils import load_pickle, pad_sentence_batch, get_batches

def placeholders():
	input_data = tf.placeholder(tf.int32, [None, None], name='input')
	targets = tf.placeholder(tf.int32, [None, None], name='targets')
	lr = tf.placeholder(tf.float32, name='learning_rate')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
	summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
	max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
	text_length = tf.placeholder(tf.int32, (None,), name='text_length')

	return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

def process_decoding_input(target_data, vocab_to_int, batch_size):
	ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1]) 
	#add to the start of each input
	dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<SOS>']), ending], 1)

	return dec_input

def make_rnn_cell(rnn_size, keep_probability):
	"""Creates LSTM cell wrapped with dropout.
	"""
	cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1))
	cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_probability)
	return cell

'''
rnn_size: The number of units in the LSTM cell
sequence_length: size [batch_size], containing the actual lengths for each of the 
				sequences in the batch
keep_prob: RNN dropout input keep probability
'''
def encoding_layer(rnn_size, sequence_length,rnn_inputs, keep_prob):
	with tf.variable_scope('encoder'):
		cell_fw = make_rnn_cell(rnn_size,keep_prob)

		cell_bw = make_rnn_cell(rnn_size,keep_prob)

		enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
																cell_bw,
																rnn_inputs,
																sequence_length,
																dtype=tf.float32)
		enc_output = tf.concat(enc_output,2)
		
	return enc_output, enc_state

'''
parameters
dec_embed_input: output of embedding_lookup for a batch of inputs
summary_length: length of each padded summary sequences in batch, since padded, all 
				lengths should be same number
dec_cell: the decoder RNN cells' output with attention wapper
output_layer: fully connected layer to apply to the RNN output
vocab_size: vocabulary size i.e. len(vocab_to_int)+1
max_summary_length: the maximum length of a summary in a batch
batch_size: number of input sequences in a batch
'''
def training_decoding_layer(dec_embed_input, summary_length, dec_cell, 
							output_layer, vocab_size, max_summary_length, batch_size):
	
	#TraingHelper reads a sequence of integers from the encoding layer.
	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
														sequence_length=summary_length,
														time_major=False)

	#BasicDecoder processes the sequence with the decoding cell, and an output layer, 
	#		which is a fully connected layer. initial_state set to zero state.
	training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
													helper=training_helper,
													initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
													output_layer = output_layer)

	#dynamic_decode creates our outputs that will be used for training.
	training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
														output_time_major=False,
														impute_finished=True,
														maximum_iterations=max_summary_length)
	return training_logits

'''
parameters
embeddings: the GloVe's word_embedding_matrix
start_token: the id of <SOS>
end_token: the id of <EOS>
dec_cell: the decoder RNN cells' output with attention wapper
output_layer: fully connected layer to apply to the RNN output
max_summary_length: the maximum length of a summary in a batch
batch_size: number of input sequences in a batch
'''
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, output_layer,
							max_summary_length, batch_size, beam_width):
	
	#Create the inference logits
	start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
	

	#Greedy Implementation	
	#GreedyEmbeddingHelper argument start_tokens: int32 vector shaped of the start tokens.
	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
																start_tokens,
																end_token)
	
	inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
														inference_helper,
														dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
														output_layer)
	
	#beam search implementation
	'''
	inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=dec_cel,
															embedding=embeddings,
															start_tokens=start_tokens,
															end_tokens=start_tokens,
															initial_state= cell.zero_state(batch_size*beam_width, tf.float32),
															beam_width=beam_width,
															output_layer=output_layer)
    '''
	inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
															output_time_major=False,
															impute_finished=True,
															maximum_iterations=max_summary_length)	

	return inference_logits

'''
parameters

dec_embed_input: output of embedding_lookup for a batch of inputs
embeddings: the GloVe's word_embedding_matrix
enc_output: encoder layer output, containing the forward and the backward rnn output
enc_state: encoder layer state, a tuple containing the forward and the backward final states of bidirectional rnn.
vocab_size: vocabulary size i.e. len(vocab_to_int)+1
text_length: the actual lengths for each of the input text sequences in the batch
summary_length: the actual lengths for each of the input summary sequences in the batch
max_summary_length: the maximum length of a summary in a batch
rnn_size: The number of units in the LSTM cell
vocab_to_int: vocab_to_int the dictionary
keep_prob: RNN dropout input keep probability
batch_size: number of input sequences in a batch
'''
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
				max_summary_length, rnn_size, decode_size,vocab_to_int, keep_prob, batch_size, beam_width, train_mode):
	
	#Create the decoding cell and attention for the training and inference decoding layers
	dec_cell = make_rnn_cell(decode_size,keep_prob)
	output_layer = Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
	attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
													enc_output,
													text_length,
													normalize=False,
													name='BahdanauAttention')
	
	#AttentionWrapper applies the attention mechanism to our decoding cell.												
	dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)
	
	with tf.variable_scope("decode"):
		training_logits = training_decoding_layer(dec_embed_input,summary_length,dec_cell,
												output_layer,
												vocab_size,
												max_summary_length,
												batch_size)

	with tf.variable_scope("decode", reuse=True):
		inference_logits = inference_decoding_layer(embeddings,
													vocab_to_int['<SOS>'],
													vocab_to_int['<EOS>'],
													dec_cell,
													output_layer,
													max_summary_length,
													batch_size,
                                                    beam_width)
	return training_logits , inference_logits


def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
				vocab_size, rnn_size, decode_size, vocab_to_int, batch_size, beam_width,train_mode):

	# Use GloVe's embeddings and the newly created ones as our embeddings
	embeddings = word_embedding_matrix

	enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
	enc_output, enc_state = encoding_layer(rnn_size, text_length, enc_embed_input, keep_prob)
	
	dec_input = process_decoding_input(target_data, vocab_to_int, batch_size)
	dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
	
	training_logits, inference_logits  = decoding_layer(dec_embed_input,
														embeddings,
														enc_output,
														enc_state,
														vocab_size,
														text_length,
														summary_length,
														max_summary_length,
														rnn_size,
														decode_size,
														vocab_to_int,
														keep_prob,
														batch_size,
														beam_width,
														train_mode)
	return training_logits, inference_logits
	

############################################################################################
#load summaries and word embeddings
int_summaries = load_pickle('./data/cnn_train_int_summaries.p')
int_texts = load_pickle('./data/cnn_train_int_texts.p')

print('total articles: '+str(len(int_texts)))


word_embedding_matrix = load_pickle('./data/cnn_word_embedding_matrix.p')
vocab_to_int = load_pickle('./data/cnn_vocab_to_int.p')
int_to_vocab = load_pickle('./data/cnn_int_to_vocab.p')

# Set the Hyperparameters
epochs = 33
batch_size = 10
rnn_size = 256
decode_size = 100
learning_rate = 0.005
keep_probability = 0.95
beam_width = 1
train_mode = True
learning_rate_decay = 0.95
min_learning_rate = 0.0005

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

	# Load the model inputs
	input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = placeholders()

	# Create the training and inference logits
	training_logits, inference_logits = seq2seq_model(input_data,
													  targets,
													  keep_prob,
													  text_length,
													  summary_length,
													  max_summary_length,
													  len(vocab_to_int)+1,
													  rnn_size,
													  decode_size,
													  vocab_to_int,
													  batch_size,
													  beam_width,
													  train_mode)

	# Create tensors for the training logits and inference logits
	training_logits = tf.identity(training_logits[0].rnn_output, name= 'logits')
	inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')

	# Create the weights for sequence_loss, the should be all True across since each batch is padded
	masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

	with tf.name_scope("optimization"):
		# Loss function
		cost = tf.contrib.seq2seq.sequence_loss(
			training_logits,
			targets,
			masks)

		# Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate)

		# Gradient Clipping
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.apply_gradients(capped_gradients)

print("Graph is built.")
graph_location = "./graph"
print(graph_location)

train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(train_graph)

#################################################################################
# Train the Model																#
#################################################################################
display_step = 20 # Check training loss after every 20 batches
stop_early = 0
stop = 5 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 7 # Make 3 update checks per epoch
update_check = (len(int_summaries)//batch_size//per_epoch)-1

update_loss = 0
batch_loss = 0
summary_update_loss = [] # Record the update losses for saving improvements in the model

checkpoint = "./checkpoint_model"
with tf.Session(graph=train_graph) as sess:
	sess.run(tf.global_variables_initializer())

	for epoch_i in range(1, epochs+1):
		update_loss = 0
		batch_loss = 0
		for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
				get_batches(int_summaries, int_texts, batch_size,vocab_to_int)):
			start_time = time.time()
			_, loss = sess.run(
				[train_op, cost],
				{input_data: texts_batch,
				 targets: summaries_batch,
				 lr: learning_rate,
				 summary_length: summaries_lengths,
				 text_length: texts_lengths,
				 keep_prob: keep_probability})

			batch_loss += loss
			update_loss += loss
			end_time = time.time()
			batch_time = end_time - start_time

			if batch_i % display_step == 0 and batch_i > 0:
				print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
					  .format(epoch_i,
							  epochs,
							  batch_i,
							  len(int_summaries) // batch_size,
							  batch_loss / display_step,
							  batch_time*display_step))
				batch_loss = 0

			if batch_i % update_check == 0 and batch_i > 0:
				print("Average loss for this update:", round(update_loss/update_check,3))
				summary_update_loss.append(update_loss)

				# If the update loss is at a new minimum, save the model
				if update_loss <= min(summary_update_loss):
					print('New Record!')
					stop_early = 0
					saver = tf.train.Saver()
					saver.save(sess, checkpoint)

				else:
					print("No Improvement.")
					stop_early += 1
					saver = tf.train.Saver()
					saver.save(sess, checkpoint)
					if stop_early == stop:
						break
				update_loss = 0


		# Reduce learning rate, but not below its minimum value
		learning_rate *= learning_rate_decay
		if learning_rate < min_learning_rate:
			learning_rate = min_learning_rate
	'''
		if stop_early == stop:
			print("Stopping Training.")
			break
	'''       
	saver = tf.train.Saver()
	saver.save(sess, checkpoint)
	print('Saved')
