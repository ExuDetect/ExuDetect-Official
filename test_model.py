import tensorflow as tf
import numpy as np

# 1. Model load karo
interpreter = tf.lite.Interpreter(model_path="eyedetecter.tflite")
interpreter.allocate_tensors()

# 2. Input aur Output details check karo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Total Outputs:", len(output_details))

# 3. Dummy data ke saath test karo (Random image)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 4. Saare 4 outputs print karo (Netron wale)
for i in range(len(output_details)):
    output_data = interpreter.get_tensor(output_details[i]['index'])
    print(f"Output {i} Result:", output_data)