from os import path
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import spearmanr

import librosa
from matplotlib import pyplot as plt
import re

import myprosody as mysp


class AudioAnalyzer:
    """
    Class used to extract insight from audio
    """

    __dir_path = path.dirname(path.realpath(__file__))
    __PROSODY_FOLDER = __dir_path + "/../../res/lib/myprosody"
    __prosody_csv_file = __PROSODY_FOLDER + "/prosody.csv"
    
    corpus_pros = pd.read_csv(__prosody_csv_file, sep=';', index_col=0)
    

    def __init__(self, datash):
        """
        Instance used to extract insights from audio
        """
        self.data = datash
        
        self.my_pros = self.prosody()
        self.save_prosody()        

    def prosody(self):
        """
        return: dict of a few essential prosodic features in a dictionnary
        """
        # MyProsody seems to print values, and doesnt return them :/
        # -> using praat directly doesnt seems easy
        # -> capture output ?
        # the lib is really fragile, and rate_of_speech and articulation_rate are rounded :/
        file_like = io.StringIO()
        with redirect_stdout(file_like):
            mysp.mysptotal(self.data.fname, AudioAnalyzer.__PROSODY_FOLDER)
        lines = file_like.getvalue().splitlines()[2:] # first 2 lines are useless 
        prosody_dict = {}
        # rename first key from 'number_ of_syllables' (space mistake) to 'number_of_syllables'
        _, value = re.split(' {2,}', lines.pop(0))
        prosody_dict['number_of_syllables'] = float(value)
        # store the other values as they are in the dict
        for line in lines:
            key, value = re.split(' +', line)
            prosody_dict[key] = float(value) # some of them are usually integer
        # rate_of_speech and articulation_rate are rounded by myprosody
        # recalculate there unrounded values
        prosody_dict['rate_of_speech'] = prosody_dict['number_of_syllables'] / prosody_dict['original_duration']
        prosody_dict['articulation_rate'] = prosody_dict['number_of_syllables'] / prosody_dict['speaking_duration']
        return prosody_dict

    def save_prosody(self):
        """
        Save prosody as a new line in a csv file (if not already in there)
        """
        my_pros_df = pd.DataFrame([list(self.my_pros.values())], columns=list(self.my_pros.keys()))
        total_pros = pd.concat([my_pros_df, AudioAnalyzer.corpus_pros])
        total_pros.drop_duplicates(subset=['number_of_syllables', 'f0_mean', 'original_duration'], inplace=True, keep='last')
        total_pros.to_csv(AudioAnalyzer.__prosody_csv_file, sep=';')
    
    def get_loudness_array(self):
        """
        return :
            time_array : timestamp of each chunk
            loudness_array : normalized array of log10(rms) for each chunk
        """
        CHUNK = 2048 # can be tuned, here it's a time step of ~0.13 secondes for a rate of 16kHz
        audio_array = self.data.get_audio(self.data.api_rate)
        rms_array = np.zeros(int(len(audio_array)/CHUNK)-1)
        # calculate RMS for each chunk
        for i in range(int(len(audio_array)/CHUNK)-1):
            current_chunk = audio_array[i*CHUNK:(i+1)*CHUNK]
            # Convert to float from int16 or int32
            rms = self.__rms_chunk(current_chunk)
            rms_array[i] = np.log10(rms) # log10 of rms actually
        # Get max to normalize
        max_rms = np.max(rms_array)
        # timestamp for each element in rms_array
        time_array = np.array([i*CHUNK/self.data.api_rate for i in range(int(len(audio_array)/CHUNK)-1)])
        return time_array, rms_array/max_rms

    def __rms_chunk(self, audio_chunk):
        """
        audio_chunk : numpy array of audio
        return :
            loudness : rms for  chunk
        """
        # calculate RMS
        # Convert to float from int16 or int32
        d = np.frombuffer(audio_chunk, audio_chunk.dtype).astype(np.float)
        rms = np.sqrt(np.mean(d**2))
        return rms

    def plot_loudness(self):
        """
        Plot loudness against time
        """
        x,y = self.get_loudness_array()
        plt.plot(x,y)
        plt.show()

    def get_sentence_timestamps(self):
        """
        DO NOT USE: sentence are now used with DataFrame rather than transcript
        return:
            numpy array of [start_time, end_time] for each sentence
        """
        sentence_array = self.data.split_sentences()
        words_df = self.data.words_df
        words = words_df[["word"]].to_numpy().flatten()
        # for now, end_times are equal to following word start_time, but it should be improved so we use both
        start_times = words_df[["start_time"]].to_numpy().flatten()
        end_times = words_df[["end_time"]].to_numpy().flatten()
        word_index = 0
        sentence_times = np.zeros((len(sentence_array), 2))

        # construct sentence word by word and check if matching
        # coud also just detect . , ? ! which are kept along side the previous word.
        for i in range(len(sentence_array)):
            start = start_times[word_index]

            proposal = words[word_index]
            while proposal in sentence_array[i]:
                word_index += 1
                if not word_index < len(words):
                    break
                proposal += ' ' + words[word_index]

            sentence_times[i] = np.array([start, end_times[word_index-1]])

        return sentence_times

    def split_between_words(self, chunk_size = 2.5, use_sentence=True):
        """
        chunk_size: max size of a chunk, in seconds
        use_sentence: bool, will furthermore split between sentence if true
        return:
            2D array of timestamp of audio fname, cut between words but shorter than chunk_size. 
            Each sentence is in a separate array inside the main one.
        """
        if use_sentence:
            grouped = self.data.words_df.groupby('sentence')
        else:
            grouped = [[0, self.data.words_df]]

        cut_array = [[] for i in range(len(grouped))]
        cut_array[0].append(0)
        last_t = 0

        text = []

        for sentence_index, df in grouped:
            end_times = df[["end_time"]].to_numpy().flatten()
            words = df[["word"]].to_numpy().flatten()
            current_text = ''
            for i, t in enumerate(end_times):
                current_text += words[i] + ' '
                if t - last_t > chunk_size:
                    cut_array[sentence_index].append(cut_array[sentence_index][-1]+int(chunk_size*0.7))
                    last_t = cut_array[sentence_index][-1]
                    text.append(current_text)
                    current_text = ''
                elif t - cut_array[sentence_index][-1] > chunk_size:
                    cut_array[sentence_index].append(last_t)
                    last_t = t
                    text.append(current_text)
                    current_text = ''
                else:
                    last_t = t
            while end_times[-1] - cut_array[sentence_index][-1] > chunk_size:
                cut_array[sentence_index].append(cut_array[sentence_index][-1]+int(chunk_size*0.7))
            cut_array[sentence_index].append(end_times[-1])

            if sentence_index+1 < len(grouped):
                cut_array[sentence_index+1].append(cut_array[sentence_index][-1])

        for i, w in enumerate(text):
            print(i,w)

        return cut_array

    def moving_average(self, data, window_width):
        """
        return:
            numpy array of moving average on 'data' using a winfows width of size 'window_width'
        """
        cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
        return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    def normalize(self, data, ref):
        """
        return:
            numpy array of data, centered around 'ref' and in [-1, 1]
        """
        data = data - ref
        return data / np.max(np.abs(data))

    def get_arousal(self, Fs=None, n_frame=512, kernel=15):
        """
        Fs: sample rate to use for analysis
        n_frame: number of frame used for each fft
        kernel: numner of frame used to compute moving median/average
        return: numpy arrays:
            - time array (in s)
            - arousal array (centered around mean value, and in [-1, 1])
        """
        if Fs is None:
            Fs = self.data.api_rate
        audio = self.data.get_audio(Fs)
        # F0
        np_f0 = librosa.yin(audio, 50, 500, Fs, n_frame)
        # Intensity
        np_int = librosa.feature.rms(audio, frame_length=n_frame,hop_length=n_frame//4)[0]
        # HF500
        np_spe = np.transpose(np.abs(librosa.stft(audio, n_frame)))
        # np_spe[frame_i, freq_j] is the magnitude of the frequence of index j at 
        # the frame of index i.
        # Conversion index-> Hz, s
        frame_index = [i for i in range(len(np_spe))]
        np_freq = librosa.fft_frequencies(Fs, n_frame)
        np_time = librosa.frames_to_time(frame_index, Fs, hop_length=n_frame//4, n_fft = n_frame)
        # Find index to use for calculting HF500
        low_index = 0 
        while np_freq[low_index] < 80:
            low_index+=1
        high_index = low_index
        while np_freq[high_index] < 500:
            high_index+=1
        # Compute HF500
        np_hf500 = np.zeros(len(np_spe))
        invalid_index = []
        for i, mag in enumerate(np_spe):
            s = sum(mag[low_index:high_index])
            if s != 0:
                np_hf500[i] = sum(mag[high_index:]) / s
            else:
                print("invalid hf500 value at {}s".format(i*n_frame/(Fs*4)))
                invalid_index.append(i)

        for i in invalid_index:
            np_time = np.delete(np_time, i, 0)
            np_freq = np.delete(np_freq, i, 0)
            np_f0 = np.delete(np_f0, i, 0)
            np_int = np.delete(np_int, i, 0)
            np_hf500 = np.delete(np_hf500, i, 0)

                
        # Extract median 
        np_f0 = medfilt(np_f0, kernel)
        np_int = medfilt(np_int, kernel)
        np_hf500 = medfilt(np_hf500, kernel)
        # Extract mean, which is used as reference
        f0_ref = np.mean(np_f0)
        int_ref = np.mean(np_int)
        hf500_ref = np.mean(np_hf500)

        # Score
        np_f0 = self.normalize(np_f0, f0_ref)
        np_int = self.normalize(np_int, int_ref)
        np_hf500 = self.normalize(np_hf500, hf500_ref)

        # Mean 
        np_f0 = self.moving_average(np_f0, kernel)
        np_int = self.moving_average(np_int, kernel)
        np_hf500 = self.moving_average(np_hf500, kernel)

        # Weight
        np_mean = np.mean([np_f0, np_int, np_hf500], axis=0)

        p_f0, _ = spearmanr(np_f0, np_mean)
        p_int, _ = spearmanr(np_int, np_mean)
        p_hf500, _ = spearmanr(np_hf500, np_mean)

        p_sum = p_f0 + p_int + p_hf500

        p_f0 /= p_sum
        p_int /= p_sum
        p_hf500 /= p_sum

        np_arousal = p_f0*np_f0 + p_int*np_int + p_hf500*np_hf500
        np_arousal = self.normalize(np_arousal, 0)

        offset = kernel // 2
        np_time = np_time[offset: -offset]

        return np_time, np_arousal

    def plot_arousal(self):
        """
        Plot arousal in function of time
        """
        x,y = self.get_arousal()
        x = self.moving_average(x, 50)
        y = self.moving_average(y, 50)
        plt.plot(x, y)
        plt.show()

    def arousal_per_sentence(self, Fs=None, n_frame=512, kernel=15):
        """
        Fs: sample rate to use for analysis
        n_frame: number of frame used for each fft
        kernel: numner of frame used to compute moving median/average
        return: numpy array of mean arousal per sentence
        """
        if Fs is None:
            Fs = self.data.api_rate
        grouped = self.data.words_df.groupby('sentence')
        end_times = grouped['end_time'].max()
        start_times = grouped['start_time'].min()
        np_time, np_arousal = self.get_arousal(Fs, n_frame, kernel)

        start_index = 0
        mean_sentence_arousal = []
        nb_sentence = len(end_times)
        for i in range(nb_sentence):
            while np_time[start_index] < start_times[i]:
                start_index += 1
            end_index = start_index
            while np_time[end_index] < end_times[i] and end_index < len(np_time) - 1:
                end_index += 1

            sentence_arousal = np_arousal[start_index : end_index]
            mean_sentence_arousal.append(np.mean(sentence_arousal, axis=0))

            start_index = end_index

        return mean_sentence_arousal
    
    def arousal_per_words(self, Fs=None, n_frame=512, kernel=15):
        """
        Fs: sample rate to use for analysis
        n_frame: number of frame used for each fft
        kernel: numner of frame used to compute moving median/average
        return: pandas DataFrame with column 'words' and a corresponding column 'arousal'
        """
        if Fs is None:
            Fs = self.data.api_rate
        df = self.data.words_df
        np_time, np_arousal = self.get_arousal(Fs, n_frame, kernel)
        max_frame = len(np_arousal)-1

        # time to frame
        ttf = lambda i: np.searchsorted(np_time, i) if np.searchsorted(np_time, i) < max_frame else max_frame
        df['start_frame'] = df['start_time'].apply(ttf)
        df['end_frame'] = df.apply(lambda x: ttf(x.end_time) if x.start_time != x.end_time else ttf(x.end_time+0.1), axis=1)
        df['arousal'] = df.apply(lambda i: np.mean(np_arousal[i.start_frame : i.end_frame]), axis=1)

        # Normalize
        df['arousal'] = (df['arousal']/df['arousal'].abs().max() +  df['arousal'])/2

        # Underline top words
        limit = df['arousal'].quantile(0.9)
        df['underline'] = df['arousal'].apply(lambda i: True if i>=limit else False)

        return df

    def format_arousal(self, Fs=None, n_frame=512, kernel=15):
        """
        Return a list of dict (one for each word) with keys :       
            - word
            - start_time
            - arousal
        """
        df = self.arousal_per_words(Fs, n_frame, kernel)
        df = df.drop(columns=['end_time', 'start_frame', 'end_frame'])
        dfg = df.groupby('sentence')
        arousal_dict = []
        for _, group in dfg:
            arousal_dict.append(group.drop(columns=['sentence']).to_dict('records'))
        return arousal_dict


    def get_top_words(self, nb_of_words=5):
        """
        nb of words: int, number of words to select
        return: 
            return an dataframe containing the 'nb_of_words' loudest words
        """
        words_df = self.data.words_df
        rate =  self.data.api_rate
        audio = self.data.audio_dict[rate]

        words_df['RMS'] = words_df.apply(
            lambda row: self.__rms_chunk(audio[int(row.start_time*rate) : int(row.end_time*rate)]),
            axis = 1)

        return (words_df.sort_values('RMS', ascending=False).head(nb_of_words))
        