import subprocess
from os import path
from os import name as os_name
from os import remove
import io

import pandas as pd
import numpy as np
import re

import librosa
import librosa.effects

from google.cloud import speech


class DataShelf:
    """
    Class taking the raw video as input, and offering audio, transcript and word timestamps as attributes
    """

    def __init__(self, fname, question, video_file_format="webm", api_sample_rate = 16000, load_from_save=True, save_to_file=True, txt_only=False):
        """
        fname: str, file name of the video, no format extension
        question: str, quesion asked for this data
        video_file_format : str, format of the video
        api_sample_rate: int, audio sample rate to use with API (Google recommends 16kHz)
        load_from_save: bool, if True, try to load audio/transcript/timestamp from saved files rather than video
        save_to_file: bool, if True, save transcript/timestamp in files
        txt_only: bool, if True, use only .txt/.csv file to initiate. WILL NOT BE COMPLETE, USE ONLY IN DEV
        """

        # Attributes
        self.fname = fname
        self.api_rate = api_sample_rate
        self.audio_dict = {}
        self.question = question
        self.transcript = ''
        self.words_df = pd.DataFrame()
        self.duration = 0

        # Privates
        self.__api_encoding_byte = 2 # we save audio for google API on 16 bits -> 2 bytes
        
        # Files Path

        DIR_PATH = path.dirname(path.realpath(__file__))
        if os_name == 'nt':
            VIDEO_FOLDER = DIR_PATH + "\\..\\..\\res\\data\\raw\\" # change to s3 bucket
            AUDIO_API_FOLDER = DIR_PATH + "\\..\\..\\res\\data\\audio\\"
            AUDIO_PROSODY_FOLDER = DIR_PATH + "\\..\\..\\res\\lib\\myprosody\\dataset\\audioFiles\\"
            TEXT_FOLDER = DIR_PATH + "\\..\\..\\res\\data\\text\\"
        else:
            VIDEO_FOLDER = DIR_PATH + "/../../res/data/raw/" # change to s3 bucket
            AUDIO_API_FOLDER = DIR_PATH + "/../../res/data/audio/"
            AUDIO_PROSODY_FOLDER = DIR_PATH + "/../../res/lib/myprosody/dataset/audioFiles/"
            TEXT_FOLDER = DIR_PATH + "/../../res/data/text/"

        self.TRANSCRIPT_PATH = "{}{}.txt".format(TEXT_FOLDER, self.fname)
        self.WORDS_DF_PATH = "{}{}.csv".format(TEXT_FOLDER, self.fname)
        self.AUDIO_API_PATH = "{}{}.wav".format(AUDIO_API_FOLDER, self.fname)
        self.AUDIO_PROSODY_PATH = "{}{}.wav".format(AUDIO_PROSODY_FOLDER, self.fname)
        self.VIDEO_PATH = "{}{}.{}".format(VIDEO_FOLDER, self.fname, video_file_format)  
        # Prosody intermediate result
        self.TEXT_GRID_PATH = "{}{}.TextGrid".format(AUDIO_PROSODY_FOLDER, self.fname)      

        # Init
        # Load Audio
        if not txt_only:
            if not load_from_save or not path.exists(self.AUDIO_API_PATH):
                self.save_audio_s2t()

            if not load_from_save or not path.exists(self.AUDIO_PROSODY_PATH):
                self.save_audio_prosody()
                    
            # Audio arrays
            self.audio_dict[self.api_rate], _ = librosa.load(self.AUDIO_API_PATH, sr=self.api_rate)
            with io.open(self.AUDIO_API_PATH, "rb") as audio_file:
                self.audio_dict["bytes"] = audio_file.read()
        else:
            print("NO AUDIO and NO VIDEO loaded !")            

        # Read/Make transcript
        if txt_only and not (path.exists(self.TRANSCRIPT_PATH) and
                               path.exists(self.WORDS_DF_PATH)):
            print("NO txt/csv files. You cannot use txt_only=True parameter.")
        else:
            if (load_from_save and
                path.exists(self.TRANSCRIPT_PATH) and
                path.exists(self.WORDS_DF_PATH)):
                print("-> READING txt/csv files")
                self.transcript, self.words_df = self.load_from_files()            
            else:
                print("-> NO txt/csv files")
                self.transcript, self.words_df = self.transcribe(save_to_file)
            
            self.sentences = self.split_sentences()
            # Add sentence number to pandas df
            self.add_sentence_number()
            # Update attributs
            self.duration = self.words_df.iloc[-1]['end_time']


    def save_audio_s2t(self):
        """
        Extract audio from video to a .wav file, to be used by Google API : linear16 encoding, 16kHz sample rate
        """
        # Google API encoding : linear16, 16kHz
        # -af lowpass=3000,highpass=60 -filter:a loudnorm
        print("-> Using ffmpeg to extract wav audio file")
        command = 'ffmpeg -i "{}" -f wav -nostdin -y -ar 16000 -filter:a loudnorm -ac 1 -acodec pcm_s16le -vn "{}"'.format(
                self.VIDEO_PATH, self.AUDIO_API_PATH,)
        subprocess.call(command, shell=True)

    def save_audio_prosody(self):
        """
        Extract audio form video to a .wav file, to be used by MyProsody : linear32 encoding, 48kHz sample rate
        """
        # MyProsody encoding :
        # Audio files must be in *.wav format, recorded at 48 kHz sample frame and 24-32 bits of resolution.
        # Initial test suggest myprosody would also work with linear16 encoding
        # We could replace duplicate audio file by a symbolic link
        # -af lowpass=3000,highpass=60 -filter:a loudnorm
        print("-> Using ffmpeg to extract wav audio file")
        command = 'ffmpeg -i "{}" -f wav -nostdin -y -ar 48000 -filter:a loudnorm -ac 1 -acodec pcm_s32le -vn "{}"'.format(
            self.VIDEO_PATH, self.AUDIO_PROSODY_PATH)
        subprocess.call(command, shell=True)

    def get_audio(self, rate):
        """
        rate : sample rate of the audio
        return:
            numpy array of audio
        """
        if rate in self.audio_dict:
            return self.audio_dict[rate]
        self.audio_dict[rate], _ = librosa.load(self.AUDIO_PROSODY_PATH, res_type='kaiser_fast', sr=rate)
        return self.audio_dict[rate]

    def split_audio(self, max_duration=60):
        """
        max_duration: float, maximum duration between cut, in secondes
        return:
            2D numpy array : array of frame intervals to cut audio on silence, in chunk of no more than max_duration
        """
        intervals = librosa.effects.split(self.audio_dict[self.api_rate], 9) # 9dB under max is considered silence
        intervals = np.concatenate([intervals, np.array([[intervals[-1][1], len(self.audio_dict[self.api_rate])-1]])]) 
        max_len = max_duration*self.api_rate
        valid_interval = []
        start = 0 
        for i in range(len(intervals)):
            if intervals[i][1] - start > max_len:
                valid_interval.append([start, intervals[i-1][1]])
                start = intervals[i-1][1]
        valid_interval.append([start, intervals[-1][1]])

        print(valid_interval)
        return valid_interval

    def split_sentences(self, max_sentence_len=30):
        """
        Split the transcript into sentence/clause using comma, point, etc...
        And split in halves until no part exceed 'max_sentence_len' words
        return:
            array of str, each element being a sentence/clause
        """
        # split on white space following , ? ! . (to keep the special character)
        res_split = re.split(r'(?<=[,\.\!\?])\s*', self.transcript) 
        # remove the last empty str (trailing whitespace in transcribe())
        if res_split[-1] == '':
            res_split.pop()
        res_cut = []
        for sentence in res_split:
            res_cut += self.__cut_sentence(sentence, max_sentence_len)
        return res_cut

    def __cut_sentence(self, sentence, max_sentence_len=30):
        word_list = sentence.split()
        if len(word_list) > max_sentence_len:
            sep = len(word_list)//2
            half_list = [word_list[:sep], word_list[sep:]]
            res = []
            for half in half_list:
                sentence = ""
                for word in half:
                    sentence += word + " "
                res += self.__cut_sentence(sentence, max_sentence_len)
            return res
        else:
            return [sentence]

    def add_sentence_number(self):
        """
        Add a column with sentence/clause (separated by a comma, point, etc...) number to pandas df 
        return:
            None
        """
        sentence = np.zeros(len(self.words_df.index))
        sentence_index = 0
        sentence_len = len(self.sentences[0].split())
        for row_index in range(len(self.words_df.index)):
            sentence[row_index] = sentence_index
            if row_index >= sentence_len-1 and sentence_index+1 < len(self.sentences):
                sentence_index += 1
                sentence_len += len(self.sentences[sentence_index].split())
        self.words_df['sentence'] = pd.Series(sentence, dtype=np.int)

    def transcribe(self, save_to_file=True):
        """
        Transcribe the audio file using Google API.
        save_to_file: bool, if True, result are saved in a .txt and a .csv file
        return:
            transcript : str, transcript of the audio
            words_pd : dataframe of words with timestamp. Columns are word, start_time; end_time
        """

        # content size ~duraction*rate*self*encoding_byte (here rate=16kHz, encoding=linear16 -> 2 bytes)
        # cut into parts of less than 60s (Google API limits, otherwise u have to use Google Cloud Storage)
        intervals = self.split_audio(60)
        # x*2+78 to adapt index :/ (librosa and io.open dont read file the same way (header, bytes))
        HEADER_SIZE = 78
        content_part = [self.audio_dict["bytes"][s*self.__api_encoding_byte+HEADER_SIZE : e*self.__api_encoding_byte+HEADER_SIZE] for s,e in intervals]

        client = speech.SpeechClient()

        config = speech.RecognitionConfig(
            # for wav file, rate and encoding are read in header automaticaly
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.api_rate,
            language_code="fr-FR",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        transcript = ""
        words_df = pd.DataFrame(columns=['word','start_time','end_time'])

        offset = 0
        end_time = 0
        for content in content_part:
            audio = speech.RecognitionAudio(content=content)
            response = client.recognize(request={"config": config, "audio": audio})
            # Each result is for a consecutive portion of the audio. Iterate through
            # them to get the transcripts for the entire audio file.
            for result in response.results:
                alternative = result.alternatives[0] # alternative 0 is the most likely
                print("Transcript: {}".format(alternative.transcript))
                print("Confidence: {}".format(alternative.confidence))
                transcript += alternative.transcript + " " # space to prevent two words from being pushed together
                                                           # modify [:-1] in split_sentence() if u remove it

                for word_info in alternative.words:
                    word = word_info.word
                    # convert deltatime to seconds (float)
                    start_time = word_info.start_time.total_seconds() + offset
                    end_time = word_info.end_time.total_seconds() + offset
                    words_df = words_df.append({'word': word, 'start_time': start_time, 'end_time': end_time}, ignore_index=True)
            offset = end_time # offset = last end_time from a batch

        if save_to_file:
            # save word and timestamp to csv
            words_df.to_csv(self.WORDS_DF_PATH, sep=';')
            # save transcript to txt
            with open(self.TRANSCRIPT_PATH, "w") as txt_file:
                txt_file.write(transcript)

        return transcript, words_df

    def load_from_files(self):
        """
        Load transcript and timestamp from .txt/.csv files
        return:
            transcript : str, transcript of the audio
            words_pd : dataframe of words with timestamp. Columns are word, start_time; end_time
        """
        with open(self.TRANSCRIPT_PATH, "r") as txt_file:
            transcript = txt_file.read()

        words_df = pd.read_csv(self.WORDS_DF_PATH, sep=';', index_col=0)
        
        return transcript, words_df

    def remove_from_disk(self):
        """
        Remove all data related to this datashelf from disk
        """
        # txt file for transcript
        if path.exists(self.TRANSCRIPT_PATH):
            remove(self.TRANSCRIPT_PATH)
        # CSV files for words timestamps
        if path.exists(self.WORDS_DF_PATH):
            remove(self.WORDS_DF_PATH)
        # Audio file sor S2T API
        if path.exists(self.AUDIO_API_PATH):
            remove(self.AUDIO_API_PATH)
        # Audio file for prosody
        if path.exists(self.AUDIO_PROSODY_PATH):
            remove(self.AUDIO_PROSODY_PATH)
        # Video file
        if path.exists(self.VIDEO_PATH):
            remove(self.VIDEO_PATH)
        # Prosody intermediate result
        if path.exists(self.TEXT_GRID_PATH):
            remove(self.TEXT_GRID_PATH)
        
