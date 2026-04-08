import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='eyedetecter.tflite')
interpreter.allocate_tensors()
op_names = set([op['op_name'] for op in interpreter._get_ops_details()])
print("Ops in this model:")
print(op_names)
