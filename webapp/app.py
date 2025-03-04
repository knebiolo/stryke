from flask import Flask

app = Flask(__name__)

@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def index():
    return "Hello from minimal Stryke test!"
