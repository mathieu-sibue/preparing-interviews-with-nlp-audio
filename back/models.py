from mongoengine import Document
from mongoengine import StringField, ReferenceField, ListField


class Questions(Document):
    question = StringField(required=True, unique=True)
    wordsToAvoid = ListField()


class Batches(Document):
    questions = ListField(ReferenceField(Questions), required=True)


class Insights(Document):
    author = StringField(required=True)
    question = ReferenceField(Questions, required=True)
    transcript = StringField(required=True)
    insights = ListField(required=True)
