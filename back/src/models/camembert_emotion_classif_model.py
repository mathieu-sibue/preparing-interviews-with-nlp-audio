import torch.nn as nn
from transformers import CamembertConfig, CamembertModel
import os


class MyCamemBERTModel(nn.Module):
    """
    Class used to define the model on which we will load weights obtained after training camembert on GoEmotions.
    """

    # anger, disgust, fear, joy, neutrality, sadness, surprise in this order in the GoEmotions dataset
    emotions = [    
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutrality",
        "sadness",
        "surprise"
    ]
    indices = range(len(emotions))
    emotions_dict = dict(zip(emotions, indices))
    num_labels = len(emotions)


    def __init__(self, pretrained_camembert_path='../../res/models/emotion_classif/camembert_base/camembert-base-init'):
        """
        uses a pretrained model (a camembert body in our case) to define our new one for multilabel emotion classification.
        """
        super(MyCamemBERTModel, self).__init__()
        config = CamembertConfig()
        dirname = os.path.dirname(__file__)
        weights_path = os.path.join(dirname, pretrained_camembert_path)
        pretrained_camembert = CamembertModel.from_pretrained(weights_path)
        self.pretrained = pretrained_camembert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()       
   
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, 
    ):
        """
        implements a forward pass of our model (for a single sequence or a batch of sequences stored in a float tensor).
        Works properly for predictions even if only input_ids are passed.
        Needs attention_mask if used during training.
        Computes loss if labels are provided along with the input_ids.
        Returns a 1-element tuple containing predictions for each emotion in a float tensor if no label is passed.
        """
        outputs = self.pretrained(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        no_logits = self.classifier(pooled_output)

        outputs = (no_logits,) + outputs[2:]  # we add to the tuple the hidden states and attention maps if they are provided

        # labels are not essential during inference... unless we want to compute the loss
        if labels is not None:
            # the loss is computed only if we feed our model the target labels, which is necessary during training, but not for all inference calls
            loss = self.loss_fct(no_logits, labels)    
            outputs = (loss,) + outputs

        # in parentheses: not always returned
        return outputs  # (loss), logits, (hidden_states), (attentions) 
