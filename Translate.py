import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
import codecs
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import RMSprop,Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#import bleu



def load_data(url , start="", end=""):
    # Full path for the data-file.
    #path = os.path.join(url, filename)
    with open(url, encoding="utf-8") as file:
        texts = [start + line.strip() + end for line in file]

    return texts



class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens

def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean
########################################################################

mark_start = 'ssss '
mark_end = ' eeee'
url_src = ("data/"
                "train.en")
url_tgr = ("data/"
                "train.vi")
num_words = 1000
data_src = load_data(url_src)
data_dest = load_data(url_tgr,start=mark_start,end=mark_end)
    
# chuan bi data de train
print("Loading data.....")
tokenizer_src = TokenizerWrap(texts=data_src,
                              padding='pre',
                              reverse=True,
                              num_words=num_words)
tokenizer_dest = TokenizerWrap(texts=data_dest,
                               padding='post',
                               reverse=False,
                               num_words=num_words)
tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
token_start = tokenizer_dest.word_index[mark_start.strip()]
token_end = tokenizer_dest.word_index[mark_end.strip()]
encoder_input_data = tokens_src
decoder_input_data = tokens_dest[:, :-1]
decoder_output_data = tokens_dest[:, 1:]

#tao encoder
#This is the input for the encoder which takes batches of integer-token sequences.
#The None indicates that the sequences can have arbitrary length.
encoder_input = Input(shape=(None, ), name='encoder_input')
# 
embedding_size = 128
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')
#
state_size = 512
encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)

encoder_output = connect_encoder()
decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
#convert decoder_output into sequences of integer-tokens
#that can be interpreted as words from our vocabulary
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

####Tao model
decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])


optimizer = RMSprop(0.001)

decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

path_checkpoint = '21_checkpoint.h5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_checkpoint,
             callback_tensorboard]


try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
    x_data = \
    {
      'encoder_input': encoder_input_data,
      'decoder_input': decoder_input_data
    }
    y_data = \
    {
      'decoder_output': decoder_output_data
    }
    validation_split =  10000 / len(encoder_input_data)
    model_train.fit(x=x_data,
                y=y_data,
                batch_size=1000,
                epochs=10,
                validation_split=validation_split,
                callbacks=callbacks)
print('Done! You can start translate now.')
def translate(input_text, true_output_text=None):
	input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
	initial_state = model_encoder.predict(input_tokens)
	#print(initial_state)
	max_tokens = tokenizer_dest.max_tokens
	shape = (1, max_tokens)
	decoder_input_data = np.zeros(shape=shape, dtype=np.int)
	#print(decoder_input_data[0, 1])
	token_int = token_start
	output_text = ''
	count_tokens = 0
	while token_int != token_end and count_tokens < max_tokens:
		decoder_input_data[0, count_tokens] = token_int
		#print(token_int)
		x_data = \
		{
		    'decoder_initial_state': initial_state,
		    'decoder_input': decoder_input_data
		}
		decoder_output = model_decoder.predict(x_data)
		#print(decocmder_output)
		token_onehot = decoder_output[0, count_tokens, :]
		token_int = np.argmax(token_onehot)
		#print(token_int)
		sampled_word = tokenizer_dest.token_to_word(token_int)
		output_text += " " + sampled_word
		#print('+'+output_text)
		count_tokens += 1
	output_tokens = decoder_input_data[0]
	print("Input text:")
	print(input_text)
	print()
	print("Translated text:")
	#print(output_text)
	#print()
	return output_text

def translate_from_file():
    src = load_data("input.txt")
    output_file = codecs.open("output.txt" ,"w","utf-8-sig")
    for x in src:
        output_txt = translate(x)
        print(output_txt)
        output_file.write(output_txt+'\n')
    output_file.close()
    print('done')
