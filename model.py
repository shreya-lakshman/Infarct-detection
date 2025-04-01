import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Conv2D,MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

# load dataset
X_train=pickle.load(open("X_train.pickle","rb"))
y_train=pickle.load(open("Y_train.pickle","rb"))
y_test=pickle.load(open("Y_test.pickle","rb"))
X_test=pickle.load(open("X_test.pickle","rb"))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test,38)


model=Sequential()

#input layer
model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


#first hidden layer
model.add(Conv2D(128,(3,3)))
model.add(Dropout(0.3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


#second hidden layer
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Activation("relu"))


#output layer
model.add(Dense(38))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=11, batch_size=50)

model.save('savedmodel')


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, 'C:\\Users\\sange\\Desktop\\Neat_Version\\DL_final_model', 'frozen_model.pb', as_text=False)
