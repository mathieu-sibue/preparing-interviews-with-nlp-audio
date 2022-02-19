from .topic_rank import extractinformation as t
import numpy as np
import os
import spacy
from database import mongo

class Keywords:
    """
    Extract keywords from a transcript
    :attribute datashelf: the datashelf to extract keywords from
    :attribute keywords: the keywords extracted from the transcript
    :attribute keywords_with_scores: keywords with the associated scores
    """

    # QUESTIONS = {
    #     "Qu’est ce qui vous donne de l’énergie dans votre journée? Pourquoi ?": {"énergie", "journée"},
    #     "Que pensez-vous apporter dans une équipe ?": {"apporter", "équipe"},
    #     "Pouvez-vous parler d’une expérience difficile que vous avez vécue dans vos études/vie professionnelle ?": {"experience", "difficile"}
    # }

    dir_path = os.path.dirname(__file__)
    dir_path.replace('\\', '/')
    relative_path = "../../res/models/spacy_french/fr_core_news_lg-2.3.0"

    model = spacy.load(dir_path + "/" + relative_path)

    def __init__(self, datashelf):
        """
        Init the instance from a datashelf.
        :param datashelf: an instance of the DataShelf class in data.datashelf
        """

        self.datashelf = datashelf

        self._get_keywords()

    def _get_keywords(self):
        """
        Get the keywords from the transcript computed by Datashelf
        """
        # Calculate the number of keywords to extract
        n_words = self._get_n_keywords()

        # Get the question keywords to avoid them during the extraction
        question_collection = mongo.get_db().questions
        words = list(question_collection.find({"question": self.datashelf.question}))
        print(words[0]['wordsToAvoid'])
        words_to_avoid = words[0]['wordsToAvoid'] 

        # Get the keywords and the score of each keyword [(word, score),...]
        self.keywords_with_scores = t.top_phrases_extraction(self.datashelf.transcript,
                                                             no_of_phrases=n_words,
                                                             words_to_avoid=words_to_avoid,
                                                             model=self.model)
        self.keywords = [word_n_score[0] for word_n_score in self.keywords_with_scores]

    def _get_n_keywords(self):
        """
        Get the number of keyword to extract from the transcript.
        Calculated from the audio length such as:
            * 0-30s = 2 keywords
            * 30-75 = 3 keywords
            * 75-120 = 4 keywords
        """
        # computation parameters
        min_n_keywords = 2
        increment_duration = 45
        increment_start = 30

        duree = self.datashelf.duration

        additional_keywords = np.floor(((duree - increment_start) / increment_duration)) + 1

        return min_n_keywords + additional_keywords
