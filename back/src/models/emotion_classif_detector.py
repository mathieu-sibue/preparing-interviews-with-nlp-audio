import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from .camembert_emotion_classif_model import MyCamemBERTModel
from ..data.open_emotion_classif_dataset import open_emotion_classif_dataset
from ..features.tokenizer_emotion_classif import EmotionClassifTokenizer


num_labels = MyCamemBERTModel.num_labels
emotions_dict = MyCamemBERTModel.emotions_dict
emotions = MyCamemBERTModel.emotions
model_dirname = os.path.dirname(__file__)
model_path = os.path.join(model_dirname, '../../res/models/emotion_classif/camembert_trained_on_GoEmotions/camemBERT_finetuned_GoEmotions_french2_dict.pt') 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def accuracy_and_pred_vects_on_each_label(labels, preds, detection_thresholds):
    """
    function used to compute accuracy between labels and preds, also used AND to store predictions
    labels and preds are expected to be pt tensors
    detection_thresholds is expected to be an array of length 7
    """
    accuracies = []
    pred_vects = []

    ones = torch.ones_like(labels[:,0])
    zeros = torch.zeros_like(labels[:,0])

    for i in range(num_labels):

        preds_col_i = torch.where(preds[:,i] > detection_thresholds[i], ones, zeros)
        result_col_i = ((labels[:,i] == preds_col_i).sum().item() / list(preds_col_i.size())[0])
        accuracies.append(result_col_i)

        pred_vects.append((labels[:,i].cpu().numpy(),preds_col_i.cpu().numpy()))

    return accuracies, pred_vects


class EmotionDetector(object):
    """
    Class used to make simple predictions on french tokenized sentences.
    No labels needed here, unless a training is executed.
    """

    model = MyCamemBERTModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))


    def __init__(self, detection_thresholds=[0.3 for _ in range(num_labels)]):
        """
        uses detection thresholds for each emotion
        """
        if len(detection_thresholds) > num_labels:
            self.detection_thresholds = detection_thresholds[:num_labels]
        elif len(detection_thresholds) < num_labels:
            self.detection_thresholds = detection_thresholds + [0.3 for _ in range(num_labels-len(detection_thresholds))]
        else:
            self.detection_thresholds = detection_thresholds


    def detect_emotions(self, input_ids):
        """
        makes detections of emotion based on the underlying camembert model, the input_ids tensor (from the tokenizer) fed to the model and the detection thresholds for each emotion.
        Method that really adds the business value to the model.
        """
        self.model.eval()
        sigmoids = nn.Sigmoid()(self.model(input_ids)[0])
        detections = []

        for i in range(sigmoids.size()[0]):
            list_sigmoids_i = sigmoids[i,:].tolist()
            detections_i = {}

            for j in range(sigmoids.size()[1]):
                if list_sigmoids_i[j] > self.detection_thresholds[j]:
                    detections_i[emotions[j]] = list_sigmoids_i[j]

            # rule-based checks to remove potentially incoherent predictions (eg: predicting that a phrase is both marked with sadness and joy)
            if (
                ('anger' in detections_i or 'sadness' in detections_i or 'fear' in detections_i) 
                and 'joy' in detections_i
            ):
                print('hello')
                if (('anger' in detections_i and detections_i['anger'] > detections_i['joy'])
                    or ('sadness' in detections_i and detections_i['sadness'] > detections_i['joy'])
                    or ('fear' in detections_i and detections_i['fear'] > detections_i['joy'])
                ):
                    detections_i.pop('joy', None)
                else:
                    detections_i.pop('anger', None)
                    detections_i.pop('fear', None)
                    detections_i.pop('sadness', None)
            
            detections.append(detections_i)

        return detections


    def train_underlying_model(
        self, 
        train_set_filename='train_ekman_french2.tsv', 
        dev_set_filename='dev_ekman_french2.tsv',
        max_len=50,
        batch_size=16,
        learning_rate=5e-5,
        warmup_proportion=0.1,
        weight_decay=1e-7,
        adam_epsilon=1e-8,
        nb_epochs=4
    ):
        """
        trains a new instance of the underlying model (with camembert body and top classifier) using provided data of the form of the GoEmotions dataset.
        Warning: because camembert was pre-trained using a French corpus, sentences in the task-specific datasets have to be in French.
        """
        # we instantiate a new model and its tokenizer so that we can train it independently of the one currently used in the static attribute of the class, 
        # and replace it after training
        new_camembert_model = MyCamemBERTModel()
        camembert_tokenizer = EmotionClassifTokenizer()
        # we check if a GPU is available for training
        if device == 'cuda':
            new_camembert_model.cuda()

        # we import and wrangle the train dataset
        train_sentences, train_labels = open_emotion_classif_dataset('train', train_set_filename)
        # we tokenize the list of sentences it contains
        train_seqs = camembert_tokenizer.tokenize(train_sentences, max_length=max_len)
        # we build a tensor dataset then a dataloader
        train_set = TensorDataset(train_seqs.input_ids, train_seqs.attention_mask, train_labels)
        train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
        # we follow the same process for the dev set if there is one
        dev_loader = None
        if dev_set_filename is not None:
            dev_sentences, dev_labels = open_emotion_classif_dataset('dev', dev_set_filename)
            dev_seqs = camembert_tokenizer.tokenize(dev_sentences, max_length=max_len)
            dev_set = TensorDataset(dev_seqs.input_ids, dev_seqs.attention_mask, dev_labels)
            dev_loader = DataLoader(dev_set, sampler=SequentialSampler(dev_set), batch_size=batch_size)        

        # we add a bit of weight decay to regularize only the weights of certain types of layers
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [      # weight decay applicable only on certains params, not all
            {'params': [p for n, p in new_camembert_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
            {'params': [p for n, p in new_camembert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # we instantiate our optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        # we will use a scheduler
        total_nb_steps = nb_epochs * len(train_loader)    # nb_epochs * nb_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_nb_steps * warmup_proportion),
            num_training_steps=total_nb_steps
        )

        # training loop
        training_stats = []

        for _ in range(nb_epochs):

            # TRAINING:
            print("")
            print('======== Epoch {:} / {:} ========'.format(_ + 1, nb_epochs))
            print('Training on '+device+'...')
            new_camembert_model.train() 

            total_train_loss = 0
            total_train_accuracy = [0 for i in range(num_labels)]
            total_train_vects = [([],[]) for i in range(num_labels)]

            for step, batch in enumerate(train_loader):

                train_batch_input_ids, train_batch_attention_masks, train_batch_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                optimizer.zero_grad()
                (loss, output) = new_camembert_model(train_batch_input_ids, train_batch_attention_masks, labels=train_batch_labels)
                loss.backward()

                total_train_loss += loss.item()

                preds = torch.sigmoid(output)

                accuracies, pred_vects = accuracy_and_pred_vects_on_each_label(train_batch_labels, preds, self.detection_thresholds)
                total_train_accuracy = [total_train_accuracy[i]+accuracies[i] for i in range(num_labels)]
                total_train_vects = [(total_train_vects[i][0]+[pred_vects[i][0]],total_train_vects[i][1]+[pred_vects[i][1]]) for i in range(num_labels)]

                # Clip the gradient norms to 1.0.
                nn.utils.clip_grad_norm_(new_camembert_model.parameters(), 1.0)  

                optimizer.step()
                scheduler.step() 

                if step%16 == 0:
                    print("Batch "+str(step+1)+"/"+str(len(train_loader))+" training loss: "+str(loss.item())[:5]+" & accuracies: "+str(accuracies))
            
            avg_train_loss = total_train_loss / len(train_loader)   
            avg_train_accuracy = [total_train_accuracy_i / len(train_loader) for total_train_accuracy_i in total_train_accuracy]
            # f1 scores computation on the training set:
            total_train_f1 = []
            for i in range(num_labels):
                labels_i = np.concatenate(total_train_vects[i][0], axis=0)
                preds_i = np.concatenate(total_train_vects[i][1], axis=0)
                total_train_f1.append(f1_score(labels_i, preds_i)) 
            
            print(" Epoch "+str(_+1)+" avg training loss: "+str(avg_train_loss))
            print(" Epoch "+str(_+1)+" avg training accuracy: "+str(avg_train_accuracy))
            print(" Epoch "+str(_+1)+" avg training f1: "+str(total_train_f1))
            

            # VALIDATION (if a dev set is provided)
            if dev_loader is not None:
                print("")
                print('Validating...')
                avg_dev_loss, avg_dev_accuracy, total_dev_f1 = self.evaluate_other_model(new_camembert_model, dev_loader)
                print(" Epoch "+str(_+1)+" avg dev loss: "+str(avg_dev_loss))
                print(" Epoch "+str(_+1)+" avg dev accuracy: "+str(avg_dev_accuracy))
                print(" Epoch "+str(_+1)+" avg dev f1: "+str(total_dev_f1))
                training_stats.append(
                    {
                        'epoch': _ + 1,
                        'Training Loss': avg_train_loss,
                        'Training Accur.': avg_train_accuracy,
                        'Training f1': total_train_f1,
                        'Valid. Loss': avg_dev_loss,
                        'Valid. Accur.': avg_dev_accuracy,
                        'Valid. f1': total_dev_f1,
                    }
                )
            else:
                training_stats.append(
                    {
                        'epoch': _ + 1,
                        'Training Loss': avg_train_loss,
                        'Training Accur.': avg_train_accuracy,
                        'Training f1': total_train_f1,
                    }
                )          
                 
        # after the training is over, we replace self.model in the class by the newly trained model
        self.model = new_camembert_model

        # we save the weights of the new model
        torch.save(self.model.state_dict(), model_path)

        return training_stats


    def evaluate_other_model(self, my_model, dev_loader):
        """
        evaluates the performance of the underlying camembert model of the class on a torch DataSet dev_set (or test_set) loaded in a torch DataLoader dev_loader (or test_loader)
        returns scores obtained with accuracy and f1_score metrics
        """
        my_model.eval()

        total_dev_accuracy = [0 for i in range(num_labels)]
        total_dev_vects = [([],[]) for i in range(num_labels)]
        total_dev_loss = 0

        for batch in dev_loader:
            
            dev_batch_input_ids, dev_batch_attention_masks, dev_batch_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            with torch.no_grad():
                (loss, output) = my_model(dev_batch_input_ids, dev_batch_attention_masks, labels=dev_batch_labels)

            total_dev_loss += loss.item()

            preds = torch.sigmoid(output)

            accuracies, pred_vects = accuracy_and_pred_vects_on_each_label(dev_batch_labels, preds, self.detection_thresholds)
            total_dev_accuracy = [total_dev_accuracy[i]+accuracies[i] for i in range(num_labels)]
            total_dev_vects = [(total_dev_vects[i][0]+[pred_vects[i][0]],total_dev_vects[i][1]+[pred_vects[i][1]]) for i in range(num_labels)]

        avg_dev_loss = total_dev_loss / len(dev_loader)
        avg_dev_accuracy = [total_dev_accuracy_i / len(dev_loader) for total_dev_accuracy_i in total_dev_accuracy]
        # f1 scores computation on the dev set:
        total_dev_f1 = []
        for i in range(num_labels):
            labels_i = np.concatenate(total_dev_vects[i][0], axis=0)
            preds_i = np.concatenate(total_dev_vects[i][1], axis=0)
            total_dev_f1.append(f1_score(labels_i, preds_i))  

        return avg_dev_loss, avg_dev_accuracy, total_dev_f1
