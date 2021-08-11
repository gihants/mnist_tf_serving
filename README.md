AI Automation Engineer Challenge
================================

SOLUTION - GIhan Samarasinghe
===========================
## Notes on the Solution

 - In the provided solution, a grpc client is implemented for scalability purpose.
 - The inference engine (detection server) and serving interface (grpc client) are seperated and kept standalone, allowing inference engine (detection engine) to be upgraded seamlessly upon the availability of new / better model, without affecting the client serving.
 - The inference engine is continously polling for requests through REST API and / or grpc interface.
 - The solution provides a way to continously train (CT) and introduce versions of the model to the model storage.

## Solution Diagram

![Solution Architecture](solution_architecture.jpg?raw=true "Solution Architecture")

## Solution Structure
This solution contains three main components:

### 1. Continous Dev / Continous Train 
Provided in sub-package *training/*

Entrypoint:  
```sh
$python training/train.py --model_path [path to save model] --epochs [number of epochs to train] --version [version number for the model]
```
This script saves a new model in *[path to save model]/[version number for the model]*

*training/Dockerfile.gpu.amd64* provides containerised training script for systems with **NVIDIA-CUDA** enabled GPUs.

### 2. Detection Server 
A standalone detection sever is implemented using **TenserFlow Serving** in *detection_server/*

*detection_server/Dockerfile.gpu.amd* containerised the solution for **NVIDIA-CUDA** enabled GPUs. (When there is no compatible GPU, the detection server will revert back to using the CPU)

Build the docker using:
```
$docker build -f detection_server/Dockerfile.gpu.amd64 --tag [detection server image name]:[version] .
```
example Docker build:
```
$docker build -f detection_server/Dockerfile.gpu.amd64 --tag tensorflow-serving-mnist:1.0 .
```
Then run the docker (with port bindings to hosts's ports which would accept requests):
```
$docker run --rm -it -v $[model directory]:/models \
-e MODEL_NAME=[model name] \
-e MODEL_PATH=/models
-p [host's port to receive grpc requests]:8500 \
-p [host's port to act as the REST API]:8501  \
--name [name for the detection server] [name for the detection server]:[version]
```
The above command will deploy the latest version of the model within *[model directory]* as the detection server.

Example Docker run:
```
docker run --rm -it -v $(pwd)/models:/models \
-e MODEL_NAME=mnist \
-e MODEL_PATH=/models \   
-p 8500:8500 \     
-p 8501:8501 \     
--name tensorflow-serving-mnist tensorflow-serving-mnist:1.0
```

## 3. GRPC
The script *grpc_server/grpc_client.py* sends requests to the **detection server** and displays the predictions and its confidence.

Entrypoint:
```
$python grpc_server/grpc_client.py   --image [path to image file]   --model [model name]   --host [host ip] --port [host's port for detection server that receives grpc requests] 
```
The above script will print the following output on the terminal:
```
Predicted Digit:  [prediction]
Predicted Confidence:  [confidence]
```

Example usage:
```
$python grpc_server/grpc_client.py   --image data/0.png   --model mnist_model   --host localhost --port 8500
```