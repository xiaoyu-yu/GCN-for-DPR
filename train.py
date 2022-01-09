import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import input_all_data
from model import gcnnmodel
import csv
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,fbeta_score,f1_score

modelNumber =1
features=['nodeDegree', 'hortonCode', 'centerDistance',
          'centerDirection','curveLength','lengthWidthRatio',
          'rectDirection']
seed = 300
np.random.seed(seed)
tf.random.set_seed(seed)
early_stopping = 100
num_supports = 1
layers = 4
num = 300
weight_decay = 0.001
batch_size = 64
epochs = 5
learning_rate = 0.01
train_feature, train_y, train_support, test_feature, test_y, test_support, test_index =\
  input_all_data(num, num_supports,readShapeFile=False,featureGroup = features)
inputFeatures_Num = train_feature[0].shape[1]
outputSize = train_y.shape[1]
def gen(feature,supports,y,batchsize):
  count=0
  while(count<len(supports)):
    sub_support = supports[count:count+batchsize]
    for i in range(len(sub_support)):
      sub_support[i] = [tf.sparse.SparseTensor( tf.cast(item[0].astype('int64'), dtype=tf.int64), item[1], item[2]) for item in sub_support[i]]
    yield [tf.constant(feature[count:count+batchsize]),tf.constant(y[count:count+batchsize]),sub_support]
    count+=batchsize
out_shp = []
all_cost = []

model = gcnnmodel(inputFeatures_Num,outputSize)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(features,labels,L):
  with tf.GradientTape() as tape:
    predictions = model(features, L)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  labels = tf.argmax(labels, axis=1)
  train_accuracy(labels, predictions)

@tf.function
def test_step(features, labels, L):
  predictions = model(features,L)
  t_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels, name=None)
  test_loss(t_loss)
  labels = tf.argmax(labels, axis=1)
  test_accuracy(labels, predictions)

@tf.function
def valid_step(features,labels, L):
  predictions = model(features,L)
  labels = tf.argmax(labels, axis=1)
  valid_accuracy(labels, predictions)

weightSavePath_test = './save_test/group'+str(modelNumber)
checkpoint = tf.train.Checkpoint(myModel=model)
manager_test = tf.train.CheckpointManager(checkpoint, directory=weightSavePath_test, max_to_keep=3)

train_cost = []
test_cost = []
train_acc = []
test_acc = []
valid_acc=[]
bestTestLoss=1000
currentEpoch=0
savaEcpoch=0
bestValidAccuracy=0
for j in range(epochs):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  f = gen(train_feature, train_support, train_y, batch_size)
  for train_f, train_label,train_L in f:
    train_step(train_f,train_label,train_L)
  f = gen(test_feature, test_support, test_y, batch_size)
  for test_f,test_label, test_L in f:
    test_step(test_f,test_label, test_L)
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  train_cost.append(train_loss.result())
  test_cost.append(test_loss.result())
  train_acc.append(train_accuracy.result())
  test_acc.append(test_accuracy.result())
  print(template.format(j + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
  if j>50 and test_loss.result().numpy()<bestTestLoss :
    bestTestLoss=test_loss.result().numpy()
    currentEpoch = j
    if j>50 and (j - savaEcpoch > 15) :
      path = manager_test.save(checkpoint_number=epochs)
      print("model saved to %s" % path)
      savaEcpoch=j
  elif (j - currentEpoch)>=50:
    currentEpoch+=30
    optimizer.lr = optimizer.lr * (1 / (1 + 0.01 * j / 100))
    print("learning_rate:",tf.keras.backend.get_value(optimizer.lr))
    if optimizer.lr<0.0001:
      break
