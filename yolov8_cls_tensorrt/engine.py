import engine as eng
import argparse
import os
from onnx import ModelProto
import sys
import tensorrt as trt
import pandas as pd


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
       config.max_workspace_size = (256 << 20) #256 megabytes for Jetson Nano
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())

       network.get_input(0).shape = shape
       
       layer_ids = []
       layer_names = []
       output_nums = []
       output_shapes = []

       i = 0
       while network.get_layer(i) != None:
           
           layer = network.get_layer(i)
           layer_ids.append(i)
           layer_names.append(layer.name)
           output_nums.append(layer.num_outputs)
           if layer.get_output(0) == None:
               output_shapes.append("None") 
           else:
               output_shapes.append(layer.get_output(0).shape)

           i += 1
 
       df = pd.DataFrame({"layer id":layer_ids,
                          "layer_name" : layer_names,
                          "output_num": output_nums,
                          "output_shape": output_shapes})
 
       df.to_csv("layer_info.csv", index=False)

       engine = builder.build_engine(network, config)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, plan_path):
   with open(plan_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        print('***********************************************************')
        print('please set followings. args[1]: model.onnx, args[2]: save dir, args[3]: save_name.plan') 
        print('***********************************************************')
    else:
        onnx_path = args[1]
        save_dir = args[2]
        save_name = args[3]
        engine_name = os.path.join(save_dir, save_name)
        model = ModelProto()
        with open(onnx_path, "rb") as f:
            model.ParseFromString(f.read())

        d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
        d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
        batch_size = 1
        shape = [batch_size , d0, d1 ,d2]
        print("**********shape: ", shape)
        #shape = [1 , 3, 64 ,64] 
        engine = eng.build_engine(onnx_path, shape= shape)
        print("***********type engine: ", type(engine))

        eng.save_engine(engine, engine_name) 
        print('******onnx model is successfully converted to plan model******')
