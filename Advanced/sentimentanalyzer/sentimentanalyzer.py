#Implementation of recurrent neural network to analyze movie reviews on IMDB with neon

#Download imdb dataset
from neon.data import IMDB

#The dataset contains huge number of words for each reviews we're gonna limit the words and truncate the length
vocab_size = 20000
max_len = 128

#passing the word limit and truncated word length into IMDB dataset
imdb = IMDB(vocab_size, max_len)

#categorizing training dataset and testing dataset
train_set = imdb.train_iter
test_set = imdb.test_iter

#Model specification
#Initialization
from neon.initializers import GlorotUniform, Uniform

init_glorot = GlorotUniform()
init_uniform = Uniform(-0.1/128, 0.1/128)

#Following are the list of layers we are gonna implement in out network
from neon.layers import LSTM, Affine, Dropout, LookupTable, RecurrentSum
from neon.transforms import Logistic, Tanh, Softmax

layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=128, init=init_uniform),
    LSTM(output_size=128, init=init_glorot, activation=Tanh(),
         gate_activation=Logistic(), reset_cells=True),
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(nout=2, init=init_glorot, bias=init_glorot, activation=Softmax())
]

#cost optimizer and callbacks
from neon.optimizers import Adagrad
from neon.transforms import CrossEntropyMulti
from neon.layers import GeneralizedCost

cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
optimizer = Adagrad(learning_rate=0.01)

from neon.callbacks import Callbacks
num_epochs = 2
fname = 'imdb_lstm_model'

callbacks = Callbacks(model, eval_set=valid_set, eval_freq=num_epochs,
                      serialize=1, save_path=fname+'.pickle')
         

#Train Model
from neon.models import Model
model = Model(layers=layers)
model.fit(train_set, optimizer=optimizer, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

#Evaluation model on the Accuracey metric
from neon.transforms import Accuracy

print "Test Accuracy - ", 100 * model.eval(test_set,metric=Accuracy())
print "Train Accuracy - ", 100 * model.eval(train_set,metric=Accuracy())

#Inference : Using the trained model to be used to perfom on new reviews, we are gonna setup a new model with a batch size of 1

#backend setup with batch size 1
from neon.backends import gen_backend
be = gen_backend(batch_size=1)

#setting up new layers with the new batch_size
layers = [
    LookupTable(vocab_size=vocab_size, embedding_dim=embedding_dim, init=init_emb),
    LSTM(hidden_size, init_glorot, activation=Tanh(),
         gate_activation=Logistic(), reset_cells=True),
    RecurrentSum(),
    Dropout(keep=0.5),
    Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
]

#setting up new model
model_new = Model(layers=layers)

#loading the weight for new model
save_path = 'labeledTrainData.tsv' + '.pickle'
model_new.load_weights(save_path)
model_new.initialize(dataset=(sentence_length,batch_size))





