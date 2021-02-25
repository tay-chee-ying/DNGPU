import tensorflow as tf
import numpy as np
from mulmaker import make_multiply
from DNGPU import _recur as recur
#from silver_DNGPU import NGPU as silver
import os
import random
directory = os.getcwd() + "/model/_pw_dngpu_128"

length_limit = 1

max_iters = 500000
hidden_size = 128
output_size = 3
batch_size = 10
dropout = 0.05
model = recur(hidden_size,output_size)
generator = make_multiply(max_len = length_limit)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
#
#Load Weights
#

#model.load_weights
def check_mat(pred,Y):
    pred = np.round(logits.numpy())
    correct_mat = np.round(np.abs(pred-Y.numpy()))
    #compute states of correct_mat
    mean_correct = np.mean(correct_mat)
    return mean_correct    
#Logging Variables
length_limit_log = []
loss_log = []
strikes = 0
#
# Train
#
for c in range(max_iters):
    if length_limit>15:
        optimizer.learning_rate = 0.0005
    if length_limit>20:
        batch_size = 5
        optimizer.learning_rate = 0.0001
    if length_limit>30:
        batch_size = 1
    random_prob = random.random()
    if random_prob<0.2:
        cmax_len = random.randint(1,length_limit)
        rand = True
    else:
        cmax_len = length_limit
        rand = False
    generator.max_len = cmax_len
    X,Y = generator.make_batch(batch_size)
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    with tf.GradientTape() as tape:
        logits = model(X, dropout = dropout)
        logits = tf.nn.softmax(logits,axis = 2)
        loss = loss_fn(Y,logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #pass_err = check_mat(logits,Y)
    if loss.numpy()<0.05 and not(rand):
        strikes += 1
    elif loss.numpy()>0.05 and not(rand):
        strikes = 0
    if strikes>20:
        length_limit+=1
        strikes = 0
    if c%100 == 0:
        pred = np.round(logits.numpy())
        correct_mat = np.round(1-np.abs(pred-Y.numpy()))
        #compute states of correct_mat
        mean_correct = np.mean(correct_mat)
        print("Iteration: {} Length: {} Loss: {} Rand: {}".format(c,length_limit,loss.numpy(),rand))
    if c%1000 == 0:
        print("Sample_error_matrix:")
        print(correct_mat[:,:,0])
        length_limit_log += [length_limit]
        loss_log += [loss.numpy()]
        np.savetxt(directory + "/length_limit_log.csv",np.array(length_limit_log),delimiter = ',')
        np.savetxt(directory + "/loss_log.csv",np.array(loss_log),delimiter = ',')
    if c%5000 == 0:
        model.save_weights(directory + "/temp_model_{}".format(c))