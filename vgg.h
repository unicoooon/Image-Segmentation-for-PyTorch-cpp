#include<torch/torch.h>
#include<iostream>
template<typename T>
class vggBlockTwoStage:public torch::nn::Module
{
    public:
        torch::nn::Conv2d conv1 = nullptr;
        torch::nn::Conv2d conv2 = nullptr;
        torch::nn::Conv2d upconv1 = nullptr;
    public:
        vggBlockTwoStage();
        T forward(T x);
};
template<typename T>
vggBlockTwoStage<T>::vggBlockTwoStage():conv1(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                        conv2(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                        upconv1(torch::nn::Conv2dOptions(64,64,4).padding(1).stride(2).transposed(new bool(true)))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("upconv1",upconv1);
}
template<typename T>
T vggBlockTwoStage<T>::forward(T x)
{
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x,2);
    x = upconv1->forward(x);
    return x;
}

template<typename T>
class vggBlockFourStage:public torch::nn::Module
{
    public:
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv3=nullptr;
        torch::nn::Conv2d conv4=nullptr;
        torch::nn::Conv2d upconv1=nullptr;
    public:
        vggBlockFourStage();
        T forward(T x);
};
template<typename T>
vggBlockFourStage<T>::vggBlockFourStage():conv1(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                          conv2(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                          conv3(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                          conv4(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                          upconv1(torch::nn::Conv2dOptions(64,64,4).padding(1).stride(2).transposed(new bool(true)))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("conv3",conv3);
    register_module("conv4",conv4);
    register_module("upconv1",upconv1);
}
template<typename T>
T vggBlockFourStage<T>::forward(T x)
{
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    x = torch::relu(conv3->forward(x));
    x = torch::relu(conv4->forward(x));
    x = torch::max_pool2d(x,2);
    x = upconv1->forward(x);
    return x;
}
template<typename T>
class vggNet:public torch::nn::Module
{
    public:
        int inch,outch;
        vggBlockTwoStage<T> *ts1 = new vggBlockTwoStage<T>();
        vggBlockTwoStage<T> *ts2 = new vggBlockTwoStage<T>();
        vggBlockFourStage<T> *fs1 = new vggBlockFourStage<T>();
        vggBlockFourStage<T> *fs2 = new vggBlockFourStage<T>();
        torch::nn::Conv2d startConv=nullptr;
        torch::nn::Conv2d fc=nullptr;
    public:
        vggNet(int in_ch,int out_ch);
        T forward(T x);
};
template<typename T>
vggNet<T>::vggNet(int in_ch,int out_ch):inch(in_ch),outch(out_ch),startConv(torch::nn::Conv2dOptions(inch,64,3).padding(1)),
                                        fc(torch::nn::Conv2dOptions(64,outch,3).padding(1))
{
    register_module("startconv1",startConv);
    register_module("fc",fc);
}
template<typename T>
T vggNet<T>::forward(T x)
{
    x = torch::relu(startConv->forward(x));
    x = ts1->forward(x);
    x = torch::relu(x);
    x = ts2->forward(x);
    x = torch::relu(x);
    x = fs1->forward(x);
    x = torch::relu(x);
    x = fs2->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    return x;
}




