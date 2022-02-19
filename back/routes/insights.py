#from flask_cors import cross_origin
import os
import boto3
from flask import Blueprint, flash, request, jsonify
from werkzeug.utils import secure_filename
from src.models.emotion_classif_detector import EmotionDetector
from src.features.tokenizer_emotion_classif import EmotionClassifTokenizer
from src.features.keywords import Keywords
from src.features.audio_analyzer import AudioAnalyzer
from src.data.datashelf import DataShelf
from models import Insights


insights_blueprint = Blueprint('insights',__name__)


ALLOWED_EXTENSIONS = {'webm'}
UPLOAD_FOLDER = '../res/data/raw'
S3_BUCKET_NAME = '###'

s3 = boto3.resource('s3')

# Checks if file is of an allowed extensions
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@insights_blueprint.route('/compute', methods=['POST'])
def compute_insights():
    """
    uses feature-engineering and predictor objects to create insights for the user given his video answer in .webm format
    """
    # checks if the post request has the file part
    if 'file' not in request.files:
        return {
            "message": "No field 'files' in the request"
        }, 400

    print(request.files)
    file = request.files['file']

    # checks if user did not upload any file
    if file.filename == '':
        return {
            "message": "File not found in the request"
        }, 400

    elif file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, UPLOAD_FOLDER + '/' + filename)
        file.save(path)
        
        fname = file.filename.rsplit('.', 1)[0]
        author = fname.split('_')[1]
        question = request.form['question']
        print(question)

        # data wrangling
        datashelf = DataShelf(fname, question)
        sentences = datashelf.split_sentences()
        print(sentences)

        # sending resources to S3 bucket
        if os.path.exists(datashelf.TRANSCRIPT_PATH):
            transcript = open(datashelf.TRANSCRIPT_PATH,'rb')  # read bytes
            s3.Bucket(S3_BUCKET_NAME).put_object(Key='transcripts/'+fname+'.txt', Body=transcript)
        if os.path.exists(datashelf.VIDEO_PATH):
            video = open(datashelf.VIDEO_PATH,'rb')     # read bytes
            s3.Bucket(S3_BUCKET_NAME).put_object(Key='videos/'+fname+'.webm', Body=video)

        # feature eng
        emotion_tokenizer = EmotionClassifTokenizer()
        ids = emotion_tokenizer.tokenize(sentences).input_ids

        # init all the models
        nlp_emotion_detector = EmotionDetector()
        nlp_keywords_extractor = Keywords(datashelf)
        audio_analyzer = AudioAnalyzer(datashelf)

        # predictions
        nlp_emotion_detections = nlp_emotion_detector.detect_emotions(ids)
        nlp_keywords = nlp_keywords_extractor.keywords
        audio_arousal_measurements = audio_analyzer.format_arousal()
        audio_prosody_measurements = audio_analyzer.prosody()

        # remove files created on the server after performing all the inferences required to extract insights
        datashelf.remove_from_disk()

        # Saving the insights to the database
        insights = Insights(
            author=author,
            question=question,
            transcript=datashelf.transcript,
            insights=[nlp_emotion_detections, nlp_keywords, audio_arousal_measurements, audio_prosody_measurements],
        )

        print(insights.save())

        return jsonify({
            "message": "Insights computed for question: " + question,
            "transcript": sentences,
            "insights": [nlp_emotion_detections, nlp_keywords, audio_arousal_measurements, audio_prosody_measurements]
        }), 200


    else:
        return {
            "message": "File in unsupported format; only the formats " + " ".join(ALLOWED_EXTENSIONS) + " are supported"
        }, 400
