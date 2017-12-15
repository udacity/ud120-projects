from flask import Flask

print('app/__init__')
myapp = Flask(__name__)
from . import views

