#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include<iostream>
#include <string>
#include "utils.h"
#include "resnet50.h"
using namespace std;
using namespace cv;
template<typename T>
class ResBlock:public torch::nn::Module
{
    public:
        torch::nn::Conv2d conv31=nullptr;
        torch::nn::Conv2d conv11=nullptr;
        torch::nn::Conv2d conv32=nullptr;
        torch::nn::Conv2d skip_con=nullptr;
    public:
        ResBlock();
        T forward(T x);
};
template<typename T>
ResBlock<T>::ResBlock():conv31(torch::nn::Conv2dOptions(10,64,3).padding(1)),
                        conv32(torch::nn::Conv2dOptions(64,128,3).padding(1)),
                        conv11(torch::nn::Conv2dOptions(128,128,3).padding(1)),
                        skip_con(torch::nn::Conv2dOptions(10,128,3).padding(1))
{
    register_module("conv31",conv31);
    register_module("conv32",conv32);
    register_module("conv11",conv11);
    register_module("skip_con",skip_con);
}
template<typename T>
T ResBlock<T>::forward(T x)
{
    T t = skip_con->forward(x);
    x = torch::relu(conv31->forward(x));
    x = torch::relu(conv32->forward(x));
    x = torch::relu(conv11->forward(x));
    x = torch::cat({x,t},1);
    return x;
}

template<typename T>
class miniNet:public torch::nn::Module
{
    public:
        int inch,outch;
        std::unordered_map<std::string,std::vector<int>> mp;
        resnet50<torch::Tensor> *bc = new resnet50<torch::Tensor>(inch,outch,mp);
        ResBlock<torch::Tensor> *res = new ResBlock<torch::Tensor>();
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d upconv1 = nullptr;
        torch::nn::Conv2d fc=nullptr;
        torch::nn::BatchNorm bn1=nullptr;
        torch::nn::Conv2d fl = nullptr;
        torch::nn::Sequential seq = nullptr;
    public:
        miniNet(int in_ch,int out_ch,std::unordered_map<std::string,std::vector<int>> chs);
        T forward(T x);
};
template<typename T>
miniNet<T>::miniNet(int in_ch,int out_ch):inch(in_ch),outch(out_ch),mp(chs),
                                          conv1(torch::nn::Conv2dOptions(inch,64,3).padding(1)),
                                          bn1(64),
                                          upconv1(torch::nn::Conv2dOptions(64,128,4).padding(1).stride(2).transposed(new bool(true))),
                                          fc(torch::nn::Conv2dOptions(128,10,3).padding(1)),
                                          seq(torch::nn::Conv2d(torch::nn::Conv2dOptions(10,128,3).padding(1)),
                                              torch::nn::Functional(torch::relu),
                                              torch::nn::Conv2d(torch::nn::Conv2dOptions(128,10,3).padding(1))),
                                          fl(torch::nn::Conv2dOptions(256,10,3).padding(1))
{
    register_module("conv1",conv1);
    register_module("bn1",bn1);
    register_module("upconv1",upconv1);
    register_module("fc",fc);
    register_module("seq",seq);
    register_module("fl",fl);
}
template<typename T>
T miniNet<T>::forward(T x)
{
    x = conv1->forward(x);
    x = torch::relu(x);
    x = bn1->forward(x);
    x = torch::max_pool2d(x,2);
    x = upconv1->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    x = seq->forward(x);
    x = res->forward(x);
    x = bc->forward(x);
    x = torch::relu(fl->forward(x));
    return torch::log_softmax(x,1);
}
void trainModel(miniNet<torch::Tensor> &net)
{
    torch::autograd::Variable train_y = torch::randint(0,10,{1,256,256});
    torch::autograd::Variable train_x = torch::unsqueeze(train_y,0);
    torch::Tensor predData;
    torch::optim::Adam optimizer(net.parameters(),0.001);
    for(int epoch=0;epoch<100;epoch++)
    {
        predData = net.forward(train_x);
        auto loss = torch::nll_loss2d(predData,torch::_cast_Long(train_y));
        std::cout<<loss<<std::endl;
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
void predict(miniNet<torch::Tensor> net)
{
    torch::autograd::Variable test = torch::unsqueeze(torch::randint(0,10,{1,256,256}),0);
    torch::Tensor out = net.forward(test).argmax(1);
    torch::Tensor predd = out.reshape({256,256}).to(torch::kFloat32);
    std::cout<<get<0>(unique(predd))<<std::endl;
    float *rr = Tensor2Float(predd);
    cv::Mat rg = cv::Mat(256,256,CV_8UC1);
    Float2Mat(rr,rg);
    cv::imwrite("/Users/yanlang/Desktop/bce.png",rg);
}
int main()
{
    miniNet<torch::Tensor> net(1,10);
    trainModel(net);
    predict(net);
    return 0;
}

