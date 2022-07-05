from openvino.inference_engine import IECore
import cv2
ie = IECore()

net = ie.read_network(model="my/saved_model.xml",weights="my/saved_model.bin")
input_layer = next(iter(net.inputs))
print(net.inputs[input_layer].shape)
n,c,h,w = net.inputs[input_layer].shape
print(n)
#ie.add_extension("user_ie_extension/cpu/build/libuser_cpu_extension.so",device_name="CPU")

exec_net = ie.load_network(network=net,device_name="HETERO:CPU",num_requests = 1)

#layer_map = ie.query_network(network=net,device_name="CPU")

#print(layer_map)

input_image = cv2.imread("crop_image/val/no_item/2022-06-13 16:51:52.874956.jpg")
print(input_image.shape)
input_image = cv2.resize(input_image, (w,h))
input_image = input_image.transpose((2,0,1))
input_image.reshape((n,c,h,w))
#print(input_image.shape)

infer_res = exec_net.infer({input_layer:input_image})

#print(infer_res)

output_layer = next(iter(net.outputs))
print(output_layer)

#print(net.outputs[output_layer].shape)

infer_req = exec_net.start_async(request_id=0,inputs={input_layer:input_image})

status=infer_req.wait()
#print(infer_req)
res = infer_req.outputs[output_layer]
print(res)
print(format(res[0][0],".5f"))
