from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import Flask, request, jsonify
from infer import SketchInfer
import numpy as np
import argparse
import os

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
    img_numpy = np.array(content['imgs']).astype(np.uint8)
    # output_params = inference_engine.infer_img(img_numpy)
    output_params = inference_engine.infer_imgs(img_numpy)
    print(output_params)
    return jsonify({"params": output_params})


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--classification_dir', type=str, required=True)
    parser.add_argument('--regress_dirs', nargs='+', default=[])
    parser.add_argument('--resnet_type', type=str, required=True)

    args = parser.parse_args()

    base_dir = args.base_dir
    regress_dirs = args.regress_dirs
    classification_dir =  args.classification_dir

    # classification_dir = os.path.join(base_dir, args.classification_dir)
    # classification_dir = os.path.join(base_dir, args.classification_dir)
    # regress_dirs = []
    # for data_dir in args.regress_dirs:
    #     regress_dirs.append(os.path.join(base_dir, data_dir))

    # inference_engine = SketchInfer.TotalInfer(classification_dir, regress_dirs)
    inference_engine = SketchInfer.TotalInfer(base_dir, classification_dir, regress_dirs, args.resnet_type)
    #inference_engine.load_model();
    # api = Api(app)
    # api.add_resource(User, "/user/<string:name>")
    # api.add_resource(ImgInfer, "/imginfer")
    #app.run(debug=True)
    app.run()
