import librosa
import numpy as np
import os
import random
import pyworld as pw
import soundcard as sc
import soundfile as sf
import pathlib


AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_22_lang/speech'

# Noise Injections
# def add_noise(data, NOISE_FACTOR):
#     noise = np.random.randn(len(data))
#     augmented_data = data + NOISE_FACTOR * noise
#     # Cast back to same data type
#     augmented_data = augmented_data.astype(type(data[0]))
#     return augmented_data

# Shifting Time
# def time_shift(data, sampling_rate, SHIFT_MAX, shift_direction):
#     shift = np.random.randint(sampling_rate * SHIFT_MAX)
#     if shift_direction == 'right':
#         shift = -shift
#     elif shift_direction == 'both':
#         direction = np.random.randint(0, 2)
#         if direction == 1:
#             shift = -shift
#     augmented_data = np.roll(data, shift)
#     # Set to silence for heading/ tailing
#     if shift > 0:
#         augmented_data[:shift] = 0
#     else:
#         augmented_data[shift:] = 0
#     return augmented_data


# Changing Pitch
def change_pitch(data, sampling_rate, PITCH_FACTOR):
    return librosa.effects.pitch_shift(data, sampling_rate, PITCH_FACTOR)

# Changing Speed
def change_speed(data, SPEED_FACTOR):
    return librosa.effects.time_stretch(data, SPEED_FACTOR)



if __name__ == "__main__":
    wav_list = []

    for root, dirs, files in os.walk(AUDIO_DIR):
        if files:
            for file in files:
                path = root.replace('cslu_22_lang', 'cslu_22_aug')
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                print(path)
                print(file)


                y, sr = librosa.load(root+'/'+file, sr=8000)

                pitch_factor = np.random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                print(pitch_factor)
                new_file = change_pitch(y, 8000, pitch_factor)
                sf.write(path + '/pitch_' + file, new_file, 8000)

                # default_speaker = sc.default_speaker()

                # default_speaker.play(y, samplerate=sr)


                speed_factor = np.random.choice([.8, .85, .9, .95, 1.05, 1.1, 1.15, 1.2])

                # print(speed_factor)
                # new_file = change_speed(y, speed_factor)
                # print(file)
                # sf.write(path+'/speed_'+file, new_file, 8000)
                # quit()





                # default_speaker.play(new_file, samplerate=sr)



                # change_pitch()

                # wav_list.append(file)


