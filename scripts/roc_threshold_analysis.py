# -*- coding: utf-8 -*-
"""ROC-THRESHOLD-ANALYSIS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17rUprtpMvMijxbGIVou_f7Y-5JhVLu5F

#IMPORT
"""

from google.colab import drive
drive.mount('/content/drive')

# import necessari
import librosa
import numpy
import sys
import os
import glob
import itertools
import re
import pickle
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.backend as K
import keras.optimizers
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Flatten, Multiply, Add, Reshape
from tqdm import tqdm
from itertools import groupby
from keras.utils import to_categorical
from sklearn import metrics
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# costanti 
ALPHA = 0.75
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
POWER = 2.0
FRAME_NUMS = 313
VAL = 0.05

"""# DATA LOADING"""

# Loading da Google Drive
train_data = numpy.load("/content/drive/MyDrive/DCASE_DATA_EXTRACTED/train/training_pump.npy")
grouped_list_by_machine_id = pickle.load( open( "/content/drive/MyDrive/DCASE_DATA_EXTRACTED/train/training_pump_grouped_list.npy", "rb" ) )

# GENERAZIONE DELLE LABELS
# One-hot encoding
label = []
choices = []
for i in range(0, len(grouped_list_by_machine_id)):
  for j in range(0, len(grouped_list_by_machine_id[i])):
    machine_id = grouped_list_by_machine_id[i][j].split('/')[7].split('_')[2]
    #print(grouped_list_by_machine_id[i][j].split('/')[7])
    random_choice = numpy.random.choice(["match", "non_match"], p = [ALPHA, 1-ALPHA]) 

    if machine_id == '00':
      if random_choice == "match":
        to_append = [1,0,0,0]
      else: 
        not_match_label = numpy.random.choice([1, 2, 3]) 
        if not_match_label == 1:
          to_append = [0,1,0,0]
        elif not_match_label == 2:
          to_append = [0,0,1,0]
        else: 
          to_append = [0,0,0,1]

    elif machine_id == '02': 

      if random_choice == "match":
        to_append = [0,1,0,0]
      else: 
        not_match_label = numpy.random.choice( [ 1, 2, 3] ) 
        if not_match_label == 1:
          to_append = [1,0,0,0]
        elif not_match_label == 2:
          to_append = [0,0,1,0]
        else: 
          to_append = [0,0,0,1]

    elif machine_id == "04":
      
      if random_choice == "match":
        to_append = [0,0,1,0]
      else: 
        not_match_label = numpy.random.choice( [ 1, 2, 3] ) 
        if not_match_label == 1:
          to_append = [1,0,0,0]
        elif not_match_label == 2:
          to_append = [0,1,0,0]
        else: 
          to_append = [0,0,0,1]

    elif machine_id == "06":
      if random_choice == "match":
        to_append = [0,0,0,1]
      else: 
        not_match_label = numpy.random.choice( [ 1, 2, 3] ) 
        if not_match_label == 1:
          to_append = [1,0,0,0]
        elif not_match_label == 2:
          to_append = [0,1,0,0]
        else: 
          to_append = [0,0,1,0]
    
    label.append(to_append) # Append della label associata a ciascuno spettrogramma
    choices.append(random_choice) # Append della choice utilizzata per associare la label
                                  # La choice sarà utile in fase di addestramento per capire che tipo di loss calcolare

# Trasformazione in numpy.array     
label = numpy.asarray(label)
choices = numpy.asarray(choices)
print(label.shape)
print(choices.shape)

# Estrazione spettrogrammi divisi per ID
id_00 = train_data[0:906]
label_00 = label[0:906]
choices_00 = choices[0:906]

id_02 = train_data[906:1811]
label_02 = label[906:1811]
choices_02 = choices[906:1811]

id_04 = train_data[1811:2413]
label_04 = label[1811:2413]
choices_04 = choices[1811:2413]

id_06 = train_data[2413:3349]
label_06 = label[2413:3349]
choices_06 = choices[2413:3349]

id_00_training, \
id_00_validation, \
label_00_train, \
label_00_validation, \
choices_00_train, \
choices_00_validation = train_test_split(id_00, label_00, choices_00, test_size=0.33, random_state=42)

id_02_training, \
id_02_validation, \
label_02_train, \
label_02_validation, \
choices_02_train, \
choices_02_validation = train_test_split(id_02, label_02, choices_02, test_size=0.33, random_state=42)

id_04_training, \
id_04_validation, \
label_04_train, \
label_04_validation, \
choices_04_train, \
choices_04_validation = train_test_split(id_04, label_04, choices_04, test_size=0.33, random_state=42)

id_06_training, \
id_06_validation, \
label_06_train, \
label_06_validation, \
choices_06_train, \
choices_06_validation = train_test_split(id_06, label_06, choices_06, test_size=0.33, random_state=42)

# Min-Max Normalization ID_00
id_00_norm = numpy.empty_like(id_00_training)
max_00 = numpy.max(id_00_training)
min_00 = numpy.min(id_00_training)
id_00_norm = (id_00_training - min_00) / (max_00-min_00)
id_00_norm_validation = (id_00_validation - min_00) / (max_00-min_00)

# Min-Max Normalization ID_02
id_02_norm = numpy.empty_like(id_02_training)
max_02 = numpy.max(id_02_training)
min_02 = numpy.min(id_02_training)
id_02_norm = (id_02_training - min_02) / (max_02-min_02)
id_02_norm_validation = (id_02_validation - min_02) / (max_02-min_02)

# Min-Max Normalization ID_04
id_04_norm = numpy.empty_like(id_04_training)
max_04 = numpy.max(id_04_training)
min_04 = numpy.min(id_04_training)
id_04_norm = (id_04_training - min_04) / (max_04-min_04)
id_04_norm_validation = (id_04_validation - min_04) / (max_04-min_04)

# Min-Max Normalization ID_06
id_06_norm = numpy.empty_like(id_06_training)
max_06 = numpy.max(id_06_training)
min_06 = numpy.min(id_06_training)
id_06_norm = (id_06_training - min_06) / (max_06-min_06)
id_06_norm_validation = (id_06_validation - min_06) / (max_06-min_06)

print("==== DATA ====")
total_training = numpy.concatenate([id_00_norm, id_02_norm, id_04_norm, id_06_norm])
print(total_training.shape)
total_validation = numpy.concatenate([id_00_norm_validation, id_02_norm_validation, id_04_norm_validation, id_06_norm_validation])
print(total_validation.shape)

print("==== LABELS ====")
total_training_label = numpy.concatenate([label_00_train, label_02_train, label_04_train, label_06_train])
print(total_training_label.shape)
total_validation_label = numpy.concatenate([label_00_validation, label_02_validation, label_04_validation, label_06_validation])
print(total_validation_label.shape)

print("==== CHOICES ====")
total_training_choices = numpy.concatenate([choices_00_train, choices_02_train, choices_04_train, choices_06_train])
print(total_training_choices.shape)
total_validation_choices = numpy.concatenate([choices_00_validation, choices_02_validation, choices_04_validation, choices_06_validation])
print(total_validation_choices.shape)

# LAYER DEFINITION

def DenseBlock(input,n):
  x = Dense(n)(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

input_Spect = Input(shape = [128, 10])
input_Label = Input(shape = [4,])

# First Branch - Encoder
m = Flatten(input_shape = [128, 10])(input_Spect)
m = DenseBlock(m, 128)
m = DenseBlock(m, 64)
m = DenseBlock(m, 32)
m = DenseBlock(m, 16)

# Second Branch - Conditioning Feed Forward Neural Network
x = Dense(16)(input_Label)
x = Activation('sigmoid')(x)
q = Dense(16)(input_Label)

# Encoded Input Conditioning
m = Multiply()([x,m])
encoded_input_conditioned = Add()([q, m]) # Input da passare al decoder

# Decoder
m = DenseBlock(encoded_input_conditioned, 128)
m = DenseBlock(m, 128)
m = DenseBlock(m, 128)
m = DenseBlock(m, 128)
m = Dense(128*10)(m)
m = Reshape((128,10),  input_shape=(128*10,))(m) # Output del modello

loss_tracker = keras.metrics.Mean(name="loss")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class CustomModel(tensorflow.keras.Model):
    @property
    def metrics(self):
        return [loss_tracker, mse_metric]

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self([x[0],x[1]], training=False)
        # Indici match
        match = tf.where ( tf.equal(x[2][:], "match") )
        # Dati match
        data_match = K.gather(y, match)
        # Separazione dei dati PREDETTI sulla base degli indici relativi a match/non_match
        # Dati match
        pred_match = K.gather(y_pred, match)

        # Update metrica
        mse_metric.update_state(data_match, pred_match)

        return {"mse": mse_metric.result()}
    
    def train_step(self, data):
          # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
          x, y = data

          # Vettore C utilizzato per il calcolo della loss in caso di non_match
          C = 5 
          # Valore di probabilità utilizzato come peso
          ALPHA = 0.75 

          # Indici match
          match = tf.where ( tf.equal(x[2][:], "match") )

          # Indici non_match
          not_match = tf.where ( tf.equal(x[2][:], "non_match") )

          # Dati match
          data_match = K.gather(y, match)

          with tf.GradientTape() as tape:
              y_pred = self([x[0],x[1]], training=True)  # Forward pass

              # Separazione dei dati PREDETTI sulla base degli indici relativi a match/non_match
              # Dati match
              pred_match = K.gather(y_pred, match)
              # Dati non match
              pred_not_match = K.gather(y_pred, not_match) 

              loss_m = K.mean(keras.losses.mean_squared_error(data_match, pred_match)) + 1e-6  # Calcolo Loss Match
              loss_nm = K.mean(keras.losses.mean_squared_error(C,pred_not_match)) + 1e-6     # Calcolo Loss Non_Match

              loss = ALPHA * loss_m + (1 - ALPHA) * loss_nm     # loss utilizzata per l'update dei pesi

          # Compute gradients
          trainable_vars = self.trainable_variables
          gradients = tape.gradient(loss, trainable_vars)

          # Update weights
          self.optimizer.apply_gradients(zip(gradients, trainable_vars))

          # Compute our own metrics
          loss_tracker.update_state(loss)
          mse_metric.update_state(y, y_pred)
          return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

opt = keras.optimizers.Adam(
    learning_rate = 0.00001,
    beta_1=0.95,
    beta_2=0.999
)

lr_metric = get_lr_metric(opt)
model = CustomModel(inputs=(input_Spect, input_Label), outputs = m)
model.compile(optimizer = opt, metrics=["mse", lr_metric])

"""# TESTING"""

# load dataset
def select_dirs(path):
    dir_path = os.path.abspath(path)
    dirs = sorted(glob.glob(dir_path))
    return dirs

def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))

def file_list_generator(target_dir, dir_name="train", ext="wav"):
    print("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
      print("errore")
    return files


def file_to_vector_array(file_name, n_mels=64, n_fft=1024, hop_length=512, power=2.0):
    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    return log_mel_spectrogram

  
def list_to_vector_array(file_list, msg="calc...", n_mels=64, n_fft=1024, hop_length=512, power=2.0, frames=10):
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx], n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=power)
       
        # vector_array = numpy.delete(vector_array,[310,311,312], axis=1)
        # vector_array = numpy.asarray(numpy.hsplit(vector_array, 31))

        if idx == 0:
            dataset = numpy.zeros((len(file_list), n_mels, frames), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset

def key_by_id(item):
  path_splitted = item.split("/")
  file_name = path_splitted[ len(path_splitted) - 1 ]
  file_name_splitted = file_name.split("_")
  machine_id = file_name_splitted = file_name_splitted[2]
  return machine_id

def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):

    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list

def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
  
    print("target_dir : {}".format(target_dir+"_"+id_name))

    normal_files = sorted(
    glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
    normal_labels = numpy.zeros(len(normal_files))
    anomaly_files = sorted(
    glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
    anomaly_labels = numpy.ones(len(anomaly_files))
    files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
    print("test_file  num : {num}".format(num=len(files)))
    if len(files) == 0:
        print("no_wav_file!!")
    print("\n========================================")

    return files, labels

target_dir = "/content/drive/MyDrive/test/pump"

performance = []
machine_type = os.path.split(target_dir)[1]
print("============== MODEL LOAD ==============")
# set model path
model_file = "/content/drive/MyDrive/models/IDCAE/pump/1/model_pump.h5"

# load model file
if not os.path.exists(model_file):
  print("{} model not found ".format(machine_type))
  sys.exit(-1)
model = tensorflow.keras.models.load_model(model_file, custom_objects={'CustomModel': CustomModel, 'mse':mse_metric, 'lr': lr_metric})
# model.summary()

machine_id_list = get_machine_id_list_for_test(target_dir)

y_true_tot = []
y_pred_tot = []

for id_str in machine_id_list:
  # load test file

  id_num = id_str.split("_")[1]

  if id_num == "00":
    match_labels = numpy.asarray([1,0,0,0])
    max = max_00
    min = min_00
  elif id_num == "02":
    match_labels = numpy.asarray([0,1,0,0])
    max = max_02
    min = min_02
  elif id_num == "04":
    match_labels = numpy.asarray([0,0,1,0])
    max = max_04
    min = min_04
  elif id_num == "06": 
    match_labels = numpy.asarray([0,0,0,1])
    max = max_06
    min = min_06

  test_files, y_true = test_file_list_generator(target_dir, id_str)

  print("\n============== BEGIN TEST FOR A MACHINE ID {id} ==============".format(id=id_num))

  y_pred = [0. for k in test_files]

  for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):

    # Estrazione spettrogramma audio test
    data = file_to_vector_array(file_path, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=POWER)

    # Normalizzazione spettrogramma di test
    data = ( data - min ) / ( max - min )

    # Estrazione delle frame 128x10
    data_splitted = numpy.zeros((61, 128, 10))
    index = 0
    i = 0
    while i < 303:
      vector_i = numpy.zeros((128,10))
      for j in range(0,128):
        vector_i[j] = data[j][i:i+10]
      data_splitted[index] = vector_i
      index += 1
      i = i+5

    # Calcolo dell'errore medio sulle frame estratte dallo spettrogramma
    elem_error = []
    for elem in data_splitted:
      predicted = model.predict([elem.reshape(1,128,10), match_labels.reshape(1,4)])

      errors = numpy.mean(numpy.square(elem - predicted), axis=1)
      elem_error.append(numpy.mean(errors))
    # Log dell'errore associato all'istanza di test
    y_pred[file_idx] = numpy.mean(elem_error)  
 # Calcolo AUC e pAUC per i dati con un certo ID_0x
  auc = metrics.roc_auc_score(y_true, y_pred)
  p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
  
  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
  #roc_auc = metrics.auc(fpr, tpr)
  argmax = numpy.argmax(tpr - fpr)

  plt.figure(dpi=300)
  plt.title('Receiver Operating Characteristic - Complessive Results')
  plt.plot(fpr, tpr, 'b')
  plt.figtext(.6, .2, 'Threshold = %0.2f' % thresholds[argmax], size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
  plt.figtext(.6, .3, 'AUC = %0.2f' % auc, size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
  plt.figtext(.6, .4, 'pAUC = %0.2f' % p_auc, size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.plot(fpr[argmax], tpr[argmax], marker='o', markersize=10, color="red")
  plt.savefig('roc_'+id_str+'.png')
  plt.show()

  y_true_tot = numpy.append(y_true_tot, y_true, axis=0)
  print(len(y_true_tot))
  y_pred_tot = numpy.append(y_pred_tot, y_pred, axis=0)
  print(len(y_pred_tot))

fpr, tpr, thresholds = metrics.roc_curve(y_true_tot, y_pred_tot)
roc_auc = metrics.auc(fpr, tpr)
roc_pauc = metrics.roc_auc_score(y_true_tot, y_pred_tot, max_fpr=0.1)

argmax = numpy.argmax(tpr - fpr)

plt.figure(dpi=300)
plt.title('Receiver Operating Characteristic - Complessive Results')
plt.plot(fpr, tpr, 'b')
plt.figtext(.6, .2, 'Threshold = %0.2f' % thresholds[argmax], size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
plt.figtext(.6, .3, 'AUC = %0.2f' % roc_auc, size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
plt.figtext(.6, .4, 'pAUC = %0.2f' % roc_pauc, size='x-large', color="white", backgroundcolor="red", fontweight="heavy")
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(fpr[argmax], tpr[argmax], marker='o', markersize=10, color="red")
plt.savefig('roc_complessive.png')
plt.show()