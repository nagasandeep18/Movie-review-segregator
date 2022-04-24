from flask import Flask, request
from flask_cors import CORS

app = Flask("Movie_review_segregator")
CORS(app)

from api import *


@app.route("/")
def index():
    return "Welcome to Movie_review_segregator API"

@app.route("/get_pred_imdb", methods=['POST', 'GET'])
def get_pred_imdb():
    if (request.method == "POST"):
        # global session

        movie_review_url = request.form['movie_review_url']
        results = pred_imdb(movie_review_url)
        print(movie_review_url)

        return {'results': results}
    else:
        return "This API accepts only POST requests"

if __name__ == '__main__':
    
    initial_training()
    # debug = False
    debug = True
    port = 7676

    app.run(
        debug=debug,
        port=port
    )