from bson import ObjectId
from flask import Blueprint, jsonify
from database import mongo
import json
from models import Questions, Batches


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder(ensure_ascii=False).default(self, o)


questions_blueprint = Blueprint('questions', __name__)


@questions_blueprint.route('/get', methods=['GET'])
def get_questions():
    """
    fetches questions from database and sends them to the frontend
    """
    question_collection = mongo.get_db().questions
    questions = list(question_collection.find())
    for k in range(len(questions)):
        questions[k]['_id'] = str(questions[k]['_id'])
    return jsonify(questions)


