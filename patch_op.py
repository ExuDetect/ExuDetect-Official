import tflite
import flatbuffers

def find_and_patch(filename):
    with open(filename, 'rb') as f:
        buf = bytearray(f.read())
        
    model = tflite.Model.GetRootAsModel(buf, 0)
    
    num_opcodes = model.OperatorCodesLength()
    
    for i in range(num_opcodes):
        op_code = model.OperatorCodes(i)
        code = op_code.BuiltinCode()
        version = op_code.Version()
        
        if code == 9 and version == 12: # FULLY_CONNECTED v12
            print(f"Found FULLY_CONNECTED v12 at index {i}!")
            
            # Find the offset of the version field
            # The tflite schema defines version as offset 8 (0-indexed fields: 4, 6, 8)
            o = op_code._tab.Offset(8)
            if o != 0:
                pos = o + op_code._tab.Pos
                # The version is an Int32 object
                current_ver = op_code._tab.Get(flatbuffers.number_types.Int32Flags, pos)
                print(f"Current version at offset {pos} is {current_ver}")
                if current_ver == 12:
                    # Patch it to 9!
                    # Little endian 9 is 0x09 0x00 0x00 0x00
                    buf[pos] = 9
                    buf[pos+1] = 0
                    buf[pos+2] = 0
                    buf[pos+3] = 0
                    print(f"Patched to 9")
                    
                    with open(filename, 'wb') as fout:
                        fout.write(buf)
                    print(f"[{filename}] successfully patched!")
                else:
                    print(f"Unexpected version value {current_ver} at offset {pos}. Aborting.")
            else:
                print("Offset for 'version' is 0, meaning it was defaulted. Aborting.")
            break

find_and_patch('eyedetecter.tflite')
