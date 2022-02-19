from flask import Flask
from flask_cors import CORS
from routes.insights import insights_blueprint
from routes.questions import questions_blueprint
from settings import MONGO_URI
from database import mongo


app = Flask(__name__)

# cors = CORS(app, origins=['http://localhost:3000'])
# app.config['CORS_HEADERS'] = 'Content-Type'
print("[MONGO_URI]", MONGO_URI)
app.config["MONGO_URI"] = MONGO_URI
app.config["MONGODB_HOST"] = MONGO_URI

mongo.init_app(app)
CORS(app)

app.register_blueprint(insights_blueprint, url_prefix='/insights')
app.register_blueprint(questions_blueprint, url_prefix='/questions')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
