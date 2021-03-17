# IMPORT NECESSARI
import librosa
import numpy
import sys
import os
import glob
from tqdm import tqdm
from itertools import groupby
import pickle

ALPHA = 0.75
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
POWER = 2.0
FRAME_NUMS = 313

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
    # 01 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    # 02 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    #log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

  
def list_to_vector_array(file_list, msg="calc...", n_mels=64, n_fft=1024, hop_length=512, power=2.0):
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx], n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), FRAME_NUMS), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset

def key_by_id(item):
  path_splitted = item.split("/")
  file_name = path_splitted[ len(path_splitted) - 1 ]
  file_name_splitted = file_name.split("_")
  machine_id = file_name_splitted = file_name_splitted[2]
  return machine_id

# load base_directory list
dirs = "/Volumes/Extreme SSD/Tesi/DATA/valve"
files = file_list_generator(dirs)

train_data = list_to_vector_array(files, msg="generate train_dataset", n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=POWER)

train_data = train_data.reshape(len(files), N_MELS, FRAME_NUMS)

print(train_data)

with open('/Volumes/Extreme SSD/Tesi/DATA/training_valve.npy', 'wb') as f:
    numpy.save(f, train_data)

# load base_directory list
grouped_list_by_machine_id = [list(v) for k,v in groupby(sorted(files), key_by_id)]

with open('/Volumes/Extreme SSD/Tesi/DATA/training_valve_grouped_list.npy', 'wb') as file_pi:
    pickle.dump(grouped_list_by_machine_id, file_pi)
