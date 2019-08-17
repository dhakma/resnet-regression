from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import Flask, request, jsonify
from resnet import SketchInfer
import numpy as np

users = [
    {
        "name": "Nicholas",
        "age": 42,
        "occupation": "Network Engineer"
    },
    {
        "name": "Elvin",
        "age": 30,
        "occupation": "Doctor"
    },
    {
        "name": "Jass",
        "age": 22,
        "occupation": "Web developer"
    }
]

app = Flask(__name__)


@app.route('/imginfer', methods=['GET', 'POST'])
def infer():
    content = request.json
    img_numpy = np.array(content['img']).astype(np.uint8)
    output_params = inference_engine.infer_img(img_numpy)
    print(output_params)
    return jsonify({"params": output_params.tolist()})


class User(Resource):
    def get(self, name):
        for user in users:
            if (name == user["name"]):
                return user, 200
        return "User not found", 404

    def post(self, name):
        parser = reqparse.RequestParser()
        parser.add_argument("age")

class ImgInfer(Resource):
    def post(self):
        #print(request.json)
        parser = reqparse.RequestParser()
        parser.add_argument("img")
        args = parser.parse_args()
        print(args)
        return "Success", 200



if __name__ == '__main__':
    global inference_engine
    inference_engine = SketchInfer('data/multi-people-path-sketch-gen-1')
    inference_engine.load_model();
    # api = Api(app)
    # api.add_resource(User, "/user/<string:name>")
    # api.add_resource(ImgInfer, "/imginfer")
    app.run(debug=True)
