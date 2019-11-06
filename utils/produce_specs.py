import matplotlib.pyplot as plt
import librosa.display

#file_name = r'/home/robert/Downloads/CNN_results/audio/results_8s_CNNmini/p3901.flac_e40.wav'
hq_file = r'/media/veracrypt5/MLP/data/magnta/download/HQ_test/p1400.wav'
lq_file = r'/home/robert/Downloads/p1400.wav_e110.wav'


y, sr = librosa.load(hq_file)
librosa.feature.melspectrogram(y=y, sr=sr)

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram.pdf')
plt.tight_layout()

plt.savefig("original_magna_4200.pdf", bbox_inches='tight')

y, sr = librosa.load(lq_file)
librosa.feature.melspectrogram(y=y, sr=sr)

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

plt.savefig("GAN_magna_4200.pdf", bbox_inches='tight')


