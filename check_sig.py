import tensorflow as tf

def check_sig(filename):
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()
    
    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n--- Inputs ---")
    for i in input_details:
        print(f"Name: {i['name']}, Shape: {i['shape']}, Index: {i['index']}, DType: {i['dtype']}")
        
    print("\n--- Outputs ---")
    for o in output_details:
        print(f"Name: {o['name']}, Shape: {o['shape']}, Index: {o['index']}, DType: {o['dtype']}")

    # Try signatures
    try:
        sigs = interpreter.get_signature_list()
        print("\n--- Signatures ---")
        print(sigs)
    except Exception as e:
        print("\n--- Signatures Error ---")
        print(e)

check_sig('eyedetecter.tflite')
