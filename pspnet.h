#include <torch/torch.h>
#include <iostream>
template<typename T>
class resBlockp:public torch::nn::Module
{
    public:
        int ch;
        torch::Tensor x1;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv3=nullptr;
        torch::nn::Conv2d skip=nullptr;
    public:
        resBlockp(int x);
        T forward(T x);
};
template<typename T>
resBlockp<T>::resBlockp(int ch):ch(ch),
                              conv1(torch::nn::Conv2dOptions(ch,ch,3).padding(1)),
                              conv2(torch::nn::Conv2dOptions(ch,ch,3).padding(1)),
                              conv3(torch::nn::Conv2dOptions(ch,(int)(ch/2),3).padding(1)),
                              skip(torch::nn::Conv2dOptions(ch,(int)(ch/2),3).padding(1))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("conv3",conv3);
    register_module("skip",skip);
}
template<typename T>
T resBlockp<T>::forward(T x)
{
    x1 = torch::relu(conv1->forward(x));
    x1 = torch::relu(conv2->forward(x1));
    x1 = torch::relu(conv3->forward(x1));
    x = torch::relu(skip->forward(x));
    x = torch::cat({x1,x},1);
    x = torch::max_pool2d(x,2);
    return x;
}
template<typename T>
class ppm:public torch::nn::Module
{
    public:
        int ch;
        torch::Tensor c1,c2,c3,c4;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv3=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv7=nullptr;
    public:
        ppm(int ch);
        T forward(T x);
};
template<typename T>
ppm<T>::ppm(int ch):ch(ch),
                    conv1(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(1)),
                    conv2(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(1)),
                    conv3(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(1)),
                    conv7(torch::nn::Conv2dOptions(ch,(int)(ch/4),7).padding(3))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("conv3",conv3);
    register_module("conv7",conv7);
}
template<typename T>
T ppm<T>::forward(T x)
{
    c1 = torch::relu(conv1->forward(x));
    c2 = torch::relu(conv2->forward(x));
    c3 = torch::relu(conv3->forward(x));
    c4 = torch::relu(conv7->forward(x));
    x = torch::cat({c1,c2,c3,c4},1);
    return x;
}
template<typename T>
class pspNet:public torch::nn::Module
{
    public:
        int inch,outch;
        std::vector<int> vec;
        torch::nn::Conv2d st=nullptr;
        torch::nn::Conv2d fc=nullptr;
        torch::nn::Conv2d upconv1=nullptr;
        torch::nn::Conv2d upconv2=nullptr;
        torch::nn::Conv2d upconv3=nullptr;
    public:
        pspNet(int inch,int outch,std::vector<int> vec);
        T forward(T x);
};
template<typename T>
pspNet<T>::pspNet(int inch,int outch,std::vector<int> vec):inch(inch),outch(outch),vec(vec),
                 st(torch::nn::Conv2dOptions(inch,vec[0],3).padding(1)),
                 upconv1(torch::nn::Conv2dOptions(vec[2],vec[2],4).padding(1).stride(2).transposed(new bool(true))),
                 upconv2(torch::nn::Conv2dOptions(vec[1],vec[1],4).padding(1).stride(2).transposed(new bool(true))),
                 upconv3(torch::nn::Conv2dOptions(vec[0],vec[0],4).padding(1).stride(2).transposed(new bool(true))),
                 fc(torch::nn::Conv2dOptions(vec[3],outch,3).padding(1))
{
    register_module("fc",fc);
    register_module("st",st);
    register_module("upconv1",upconv1);
    register_module("upconv2",upconv2);
    register_module("upconv3",upconv3);
}
template<typename T>
T pspNet<T>::forward(T x)
{
    x = torch::relu(st->forward(x));
    x = (new resBlockp<T>(vec[0]))->forward(x);
    x = torch::relu(x);
    x = (new resBlockp<T>(vec[1]))->forward(x);
    x = torch::relu(x);
    x = (new resBlockp<T>(vec[2]))->forward(x);
    x = torch::relu(x);
    x = (new ppm<T>(vec[3]))->forward(x);
    x = torch::relu(x);
    x = upconv1->forward(x);
    x = torch::relu(x);
    x = torch::relu(upconv2->forward(x));
    x = upconv3->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    return x;
}











