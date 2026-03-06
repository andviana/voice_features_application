# import librosa
# import numpy as np

# y, sr = librosa.load("data/audio_processed/HC_AH/AH_325A_3EB21DC7-C340-4D0E-AC9E-0EABF217BBEE.wav", sr=None)

# f0 = librosa.yin(
#     y,
#     sr=sr,
#     fmin=75,
#     fmax=300
# )

# print(np.isnan(f0).sum(), "frames sem F0")

import librosa
import numpy as np
from extract_features.tsallis_f0_hist import f0_histogram_distribution, tsallis_entropy

wav = "data/audio_processed/HC_AH/AH_325A_3EB21DC7-C340-4D0E-AC9E-0EABF217BBEE.wav"
y, sr = librosa.load(wav, sr=None)

f0 = librosa.yin(y, sr=sr, fmin=100, fmax=600)
print("NaNs:", np.isnan(f0).sum(), "/", f0.size)
print("min/max:", float(np.min(f0)), float(np.max(f0)))
print("std:", float(np.std(f0)))

p = f0_histogram_distribution(y, sr, fmin_hz=75, fmax_hz=300, n_bins=50)
print("bins nonzero:", int(np.sum(p > 0)))
print("max bin prob:", float(np.max(p)))

print("shannon:", tsallis_entropy(p, q=1.0))
print("tsallis:", tsallis_entropy(p, q=1.3))