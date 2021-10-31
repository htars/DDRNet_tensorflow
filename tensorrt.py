# reference
# https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/

from tensorflow.python.compiler.tensorrt import trt_convert as trt


if __name__ == "__main__":
    input_saved_model_dir = "saved_model"
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
    converter.convert()
    output_saved_model_dir = "tensorrt_saved_model/"
    converter.save(output_saved_model_dir)
