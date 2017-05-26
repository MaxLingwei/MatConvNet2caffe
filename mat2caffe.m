clear all;

addpath /home/smartclassroom/caffe/matlab;

addpath /home/smartclassroom/Documents/tiny/matconvnet;
addpath /home/smartclassroom/Documents/tiny/matconvnet/matlab;

mat_model = './hr_res101.mat';

caffe_model = './ResNet-mat2caffe.prototxt';
caffe_net = caffe.Net(caffe_model, 'test');

caffe.set_device(0);
caffe.set_mode_gpu();


mat_net = load(mat_model);
mat_net = dagnn.DagNN.loadobj(mat_net.net);

for i = 1:length(mat_net.layers)
    layer = mat_net.layers(i);
    if strcmp(class(mat_net.layers(i).block), 'dagnn.Conv')
        paramindex = layer.paramIndexes;
        if length(paramindex) == 1
            data = mat_net.params(paramindex).value;
            caffe_net.layers(layer.name).params(1).set_data(data);
        else
            for j = 1:length(paramindex)
                data = mat_net.params(paramindex(j)).value;
                if j == 2
                    data = data';
                end
                caffe_net.layers(layer.name).params(j).set_data(data);
            end
        end
    elseif strcmp(class(mat_net.layers(i).block), 'dagnn.BatchNorm')
        params = mat_net.layers(i).params;
        bn_name = layer.name;
        scale_name = sprintf('scale%s', bn_name(3:end));

        for j = 1:length(mat_net.params)
            if strcmp(params(1), mat_net.params(j).name)
                data = mat_net.params(j).value;
                caffe_net.layers(scale_name).params(1).set_data(data);
            end
            if strcmp(params(2), mat_net.params(j).name)
                data = mat_net.params(j).value;
                caffe_net.layers(scale_name).params(2).set_data(data);
            end
        end
        for j = 1:length(mat_net.params)
            if strcmp(params(3), mat_net.params(j).name)
                data = mat_net.params(j).value;
                caffe_net.layers(layer.name).params(1).set_data(data(:, 1));
                caffe_net.layers(layer.name).params(2).set_data(data(:, 2) .^ 2);
            end
        end
        caffe_net.layers(layer.name).params(3).set_data(1.0);

    elseif strcmp(class(mat_net.layers(i).block), 'dagnn.ConvTranspose')
        data = mat_net.params(layer.paramIndexes).value;
        caffe_net.layers(layer.name).params(1).set_data(data);
    end
end

caffe_net.save('matconvnet2caffe.caffemodel');
