from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import os
os.environ['TF_KERAS'] = '1'
import onnx
import keras2onnx

onnx_model_name = 'mask-inception-onnx.onnx'

model = load_model('frsapp/models/xception.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)