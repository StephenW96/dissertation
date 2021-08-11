import librosa
import numpy as np
import os
import random
import pyworld as pw
import soundcard as sc
import soundfile as sf
import pathlib


AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_22_lang/speech'


# Changing Pitch
def change_pitch(data, sampling_rate, PITCH_FACTOR):
    return librosa.effects.pitch_shift(data, sampling_rate, PITCH_FACTOR)

# Changing Speed
def change_speed(data, SPEED_FACTOR):
    return librosa.effects.time_stretch(data, SPEED_FACTOR)

# def pitch_world(x, fs, PITCH_FACTOR):
#     x = x.astype(np.double)
#     _f0, t = pw.dio(x, fs)  # raw pitch extractor
#     f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
#     sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
#     ap = pw.d4c(x, f0, t, fs)  # extract aperiodicity
#
#     y = pw.synthesize(f0, sp, ap, fs)  # synthesize an utterance using the parameters
#     return y


if __name__ == "__main__":
    wav_list = []

    for root, dirs, files in os.walk(AUDIO_DIR):
        if files:
            for file in files:
                path = root.replace('cslu_22_lang', 'cslu_22_aug')
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                print(path)
                print(file)

                pitch_factor = np.random.choice([-4,-3,-2,-1,1,2,3,4])
                # print(pitch_factor)
                y, sr = librosa.load(root+'/'+file, sr=8000)

                # default_speaker = sc.default_speaker()

                # default_speaker.play(y, samplerate=sr)
                # new_file = change_pitch(y, 8000, pitch_factor)

                speed_factor = np.random.choice([.8, .85, .9, .95, 1.05, 1.1, 1.15, 1.2])

                print(speed_factor)
                new_file = change_speed(y, speed_factor)
                print(file)
                sf.write(path+'/speed_'+file, new_file, 8000)
                quit()





                # default_speaker.play(new_file, samplerate=sr)



                # change_pitch()

                # wav_list.append(file)


