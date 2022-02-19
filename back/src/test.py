from data.datashelf import DataShelf
from features.audio_analyzer import AudioAnalyzer
import pandas as pd 

pd.set_option('display.max_rows', 1000)

b = DataShelf('benoit-energie', 'energie', load_from_save=True)
#print(b.sentences)
#print(b.words_df)
# n = DataShelf('benoit-exp', 'energie', load_from_save=True)
# t = DataShelf('tangouette_exp', 'experience', load_from_save=True)
#print(n.sentences)
ab = AudioAnalyzer(b)
print(ab.arousal_per_words())
#print(ab.format_arousal())
#print(ab.prosody())
#print(ab.prosody())
#print(ab.arousal_per_words())
#arous = ab.arousal_per_sentence()
#for i,s in enumerate(arous):
#    print(i ,s)
#print(ab.get_emotion_df(use_sentence=False))
#for i,s in enumerate(b.sentences):
#    print(i ,s)
#ab.plot_arousal()
#print(ab.my_pros)
#print(ab.get_emotion_df())
#print(ab.format_emotion())
