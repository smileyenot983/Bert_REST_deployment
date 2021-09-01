from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from predictor import Predictor

app = Flask(__name__)
api = Api(app)



bert_predictor = Predictor()





class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        sentence = posted_data['text']

        # threshold = 0.6(86% acc)

        pred_class, pred_score = bert_predictor.predict(sentence)
        

        return jsonify({
            'Class': pred_class,
            'Score': pred_score
        })

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run('127.0.0.1',port=5500,debug=True)