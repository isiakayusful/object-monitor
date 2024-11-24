import cv2
import torch
from flask import Flask, Response, render_template


app = Flask(__name__)

def model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
