import argparse
import time
import numpy as np
import cv2

import grpc
from tensorflow import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def run(host: str, port: int, image: str, model: str):

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
	[]
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Read an image as gryescale
    data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    data = data.astype(np.float32)/255.0

    # Call classification model to make prediction on the image
    request = predict_pb2.PredictRequest()

    request.model_spec.name = model

    request.inputs['flatten_input'].CopyFrom(make_tensor_proto(data, shape=[1, 28, 28, 1]))

    result = stub.Predict(request, 10.0)

    predictions = result.outputs['dense'].float_val

    predictions = np.array(predictions)

    print("Predicted Digit: ", predictions.argmax(axis = 0))

    print("Predicted Confidence: ", max(predictions))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)

    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)

    parser.add_argument('--image', help='input image', type=str)

    parser.add_argument('--model', help='model name', type=str)

    args = parser.parse_args()

    run(args.host, args.port, args.image, args.model)