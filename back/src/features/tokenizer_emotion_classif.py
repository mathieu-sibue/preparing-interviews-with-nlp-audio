from transformers import CamembertTokenizer
import os


tokenizer_dirname = os.path.dirname(__file__)
tokenizer_path = os.path.join(tokenizer_dirname, '../../res/models/emotion_classif/camembert_base/camembert-base-tokenizer')


class EmotionClassifTokenizer(object):
    """
    Class used to tokenize french sentences transcribed from audio using camemBERT tokenizer
    """

    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)

    def __init__(self):
        """
        uses camembert tokenizer from huggingface in all cases
        """
        """
        dirname = os.path.dirname(__file__)
        tokenizer_path = os.path.join(dirname, '../../models/emotion_classif/camembert_base/camembert-base-tokenizer')
        camembert_tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = camembert_tokenizer
        """
        pass


    def tokenize(self, sentence_list, max_length=50):
        """
        uses camembert tokenizer to tokenize sentences, pad them (according to max_len) and truncate them if too long for camembert.
        The default value of the param max_len comes from monologg github repo (optimal hyperparam value of the model allegedly)
        """
        return self.tokenizer(sentence_list, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
