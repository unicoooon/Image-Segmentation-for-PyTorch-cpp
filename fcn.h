#include <torch/torch.h>
#include <iostream>
template<typename T>
class fcn:public torch::nn::Module
{
    public:
        std::vector<int> vec;
        int in_ch,out_ch;
        torch::nn::Conv2d st=nullptr;
        torch::nn::Conv2d fc=nullptr;
        torch::nn::Conv2d conv11=nullptr;
        torch::nn::Conv2d conv21=nullptr;
        torch::nn::Conv2d conv31=nullptr;
        torch::nn::Conv2d conv32=nullptr;
        torch::nn::Conv2d conv33=nullptr;
        torch::nn::Conv2d conv41=nullptr;
        torch::nn::Conv2d conv42=nullptr;
        torch::nn::Conv2d conv43=nullptr;
        torch::nn::Conv2d upconv1=nullptr;
        torch::nn::Conv2d upconv2=nullptr;
        torch::nn::Conv2d upconv3=nullptr;
    public:
        fcn(int inch,int outch,std::vector<int> vec):in_ch(inch),out_ch(outch),vec(vec),
                                                     st(torch::nn::Conv2dOptions(inch,vec[0],3).padding(1)),
                                                     conv11(torch::nn::Conv2dOptions(vec[0],vec[0],3).padding(1)),
                                                     conv21(torch::nn::Conv2dOptions(vec[1],vec[1],3).padding(1)),
                                                     conv31(torch::nn::Conv2dOptions(vec[2],vec[2],3).padding(1)),
                                                     conv32(torch::nn::Conv2dOptions(vec[2],vec[2],3).padding(1)),
                                                     conv33(torch::nn::Conv2dOptions(vec[2],vec[2],3).padding(1)),
                                                     conv41(torch::nn::Conv2dOptions(vec[3],vec[3],3).padding(1)),
                                                     conv42(torch::nn::Conv2dOptions(vec[3],vec[3],3).padding(1)),
                                                     conv43(torch::nn::Conv2dOptions(vec[3],vec[3],3).padding(1)),
                    
                  upconv1(torch::nn::Conv2dOptions(vec[3],vec[3],4).padding(1).stride(2).transposed(new bool(true))),
                  upconv2(torch::nn::Conv2dOptions(vec[2],vec[2],4).padding(1).stride(2).transposed(new bool(true))),                    upconv3(torch::nn::Conv2dOptions(vec[1],vec[1],4).padding(1).stride(2).transposed(new bool(true))),
                                                     fc(torch::nn::Conv2dOptions(vec[1],outch,3).padding(1))
{
    register_module("conv11",conv11);
    register_module("conv21",conv21);
    register_module("conv31",conv31);
    register_module("conv32",conv32);
    register_module("conv33",conv33);
    register_module("conv41",conv41);
    register_module("conv42",conv42);
    register_module("conv43",conv43);
    register_module("upconv1",upconv1);
    register_module("upconv2",upconv2);
    register_module("upconv3",upconv3);
    register_module("fc",fc);
    register_module("st",st);
}
        T forward(T x);
};
template<typename T>
T fcn<T>::forward(T x)
{
    x = torch::relu(st->forward(x));
    x = torch::relu(conv11->forward(x));
    x = torch::max_pool2d(x,2);
    x = torch::relu(conv21->forward(x));
    x = torch::max_pool2d(x,2);
    x = torch::relu(conv31->forward(x));
    x = torch::relu(conv32->forward(x));
    x = torch::relu(conv33->forward(x));
    x = torch::max_pool2d(x,2);
    x = torch::relu(conv41->forward(x));
    x = torch::relu(conv42->forward(x));
    x = torch::relu(conv43->forward(x));
    x = torch::max_pool2d(x,2);
    x = torch::relu(upconv1->forward(x));
    x = torch::relu(upconv2->forward(x));
    x = torch::relu(upconv3->forward(x));
    x = torch::relu(fc->forward(x));
    return x;
}
