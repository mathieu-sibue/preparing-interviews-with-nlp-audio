from src.models import emotion_classif_detector
import boto3
from src.features import tokenizer_emotion_classif
import pke.data_structures


sentence_list = [
    "Alors une expérience difficile dans votre vie professionnelle, je dirais que l'expérience la plus difficile à laquelle j'ai fait face c'est ma première expérience.",
    "Mas toute première fois dans ma vie professionnelle, c'était un stage allons donc stage d'été de moi dans une banque dans la finance, ça a été un gros choc parce que je sortais de centrale ou les avait des journée assez light, il y avait beaucoup de vie associative pas mal de vie chez Steve.", 
    "Je faisais beaucoup de musique et là je me suis retrouvé en costard dans des grandes tours à Londres à travailler 10h par jour et à toi rien faire d'autre de mes journées, tu que travailler donc de 8h à 20h, c'était beau et qu'il y avait les transport entre les deux et puis le reste du temps une heure pour manger ou 2h tranquille le soir mais je suis j'arrive à rien faire d'autre.",
    "C'est un gros choc comment ils sont.",
    "ça m'a presque dégoûté du monde du travail",
    "Mais après, j'ai réussi à trouver une réponse à ce problème là où je suis retourné dans la même entreprise dans la même banque à Londres pour 6 mois cette fois-ci, c'est en février 2020 et or comme quoi ça n'est pas vraiment dégoûté du monde du travail",
    "et et cette fois-ci.  J'étais beaucoup plus discipliné dans ma façon de faire ça veut dire que j'arrivais.  Avoir la discipline de me lever très tôt le matin pour faire beaucoup de sport avant le boulot ensuite, j'arrive à enchaîner avec la grosse journée de travail sans être trop fatigué et ça c'était mon rythme en semaine et le weekend, j'arrive à sortir et avoir une vie assez festif quand même",
    "et mon expérience la plus difficile, ça a donc été le choc de la façon de faire dans le monde du travail et ma réponse était avoir la discipline de trouver un mode de vie et il y a."
]

tokenizer = tokenizer_emotion_classif.EmotionClassifTokenizer()
detector = emotion_classif_detector.EmotionDetector()

input_ids = tokenizer.tokenize(sentence_list).input_ids
detections = detector.detect_emotions(input_ids)
print(detections)


#detector.train_underlying_model()