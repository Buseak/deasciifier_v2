from flask import Flask, json, request, json
import deasciifier
app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def deasciify():
    json_data = json.loads(request.data)
    deasciifier_instance = deasciifier.Deasciifier()
    response=deasciifier_instance.deasciify(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response





if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)
