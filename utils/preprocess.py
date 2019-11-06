from tqdm import tqdm
import librosa
import os
import numpy as np

hq_train_folder = r'/media/veracrypt5/MLP/data/magnta/download/HQ'
lq_train_folder = r'/media/veracrypt5/MLP/data/magnta/download/LQ_8000'
hq_test_folder = r'/media/veracrypt5/MLP/data/magnta/download/HQ_test'
lq_test_folder = r'/media/veracrypt5/MLP/data/magnta/download/LQ_test'
serialized_train_folder = r'/media/veracrypt5/MLP/data/magnta/download/genre_serial'
serialized_test_folder = r'/media/veracrypt5/MLP/data/magnta/download/genre_serial_test'

def process_and_serialize(data_type):

    """

    Serialise the data and save in separate folder.

    """

    if data_type == 'train':
        
        hq_folder = hq_train_folder

        lq_folder = lq_train_folder

        serialized_folder = serialized_train_folder

    else:

        hq_folder = hq_test_folder

        lq_folder = lq_test_folder

        serialized_folder = serialized_test_folder

    if not os.path.exists(serialized_folder):

        os.makedirs(serialized_folder)



    # walk through the path, slice the audio file, and save the serialized result

    for root, dirs, files in os.walk(hq_folder):

        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize {} audios'.format(data_type)):

            hq_file = os.path.join(hq_folder, filename)
            hq_flac, sr = librosa.load(hq_file, sr = None)
            np.save(os.path.join(hq_folder, '{}'.format(filename)), arr = hq_flac)
            filename_lq = filename.replace(".", "_lq.")
            lq_file = os.path.join(lq_folder, filename_lq)
            lq_flac, sr = librosa.load(lq_file, sr = 8000)
            np.save(os.path.join(lq_folder, '{}'.format(filename_lq)), arr = lq_flac)
            
            pair = np.array((hq_flac, lq_flac))
            np.save(os.path.join(serialized_folder, '{}'.format(filename)), arr=pair)

            
        """ A simple function to gather data about the serialized audio. Can be ignored"""
            
def test_filetype():
    serialized_folder = serialized_train_folder
    for root, dirs, files in os.walk(serialized_folder):

        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialized {} audios'.format('train')):
            file = np.load(os.path.join(serialized_folder, '{}'.format(filename)))


if __name__ == '__main__':
    process_and_serialize('train')
    process_and_serialize('test')
    #test_filetype()
    print("Dataset Loaded")
