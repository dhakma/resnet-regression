from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import Flask, request, jsonify

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
    print(content)
    return jsonify({"uuid":"success"})


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
    # api = Api(app)
    # api.add_resource(User, "/user/<string:name>")
    # api.add_resource(ImgInfer, "/imginfer")
    app.run(debug=True)
