# MatConvNet2caffe
The code can convert a matconvnet model to caffemodel.
1.You need to write a prototxt file by hand, which contains the structure of the network. 
2.Make sure you have installed the matcaffe correctly.
3.Run the matlab script.

Some details:
1.Matconvnet combines the BatchNorm layer with Scale layer, which are separated in caffe.
2.Also in BatchNorm in matconvnet, it saves the params sigma. But in caffe, it saves the params sigma ^ 2.
