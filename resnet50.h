#include<torch/torch.h>
#include<iostream>
#include<unordered_map>
#include<vector>
#include<string>
template<typename T>
class resblock:public torch::nn::Module
{
    public:
        std::unordered_map<std::string,std::vector<int>> channels;
        torch::nn::Conv2d conv3x31=nullptr;
        torch::nn::Conv2d conv3x32=nullptr;
        torch::nn::Conv2d conv1x1=nullptr;
        torch::nn::Conv2d connection=nullptr;
    public:
        resblock(std::unordered_map<std::string,std::vector<int>> ch);
        T forward(T x);   
};
template<typename T>
resblock<T>::resblock(std::unordered_map<std::string,std::vector<int>> ch):channels(ch),
conv3x31(torch::nn::Conv2dOptions(channels["conv331"][0],channels["conv331"][1],3).padding(1)),
conv3x32(torch::nn::Conv2dOptions(channels["conv332"][0],channels["conv332"][1],3).padding(1)),
conv1x1(torch::nn::Conv2dOptions(channels["conv11"][0],channels["conv11"][1],3).padding(1)),
connection(torch::nn::Conv2dOptions(channels["conn"][0],channels["conn"][1],3).padding(1))
{
    register_module("conv3x31",conv3x31);
    register_module("conv3x32",conv3x32);
    register_module("conv1x1",conv1x1);
    register_module("connection",connection);
}
template<typename T>
T resblock<T>::forward(T x)
{
    T c = connection->forward(x);
    x = conv3x31->forward(x);
    x = torch::relu(x);
    x = conv1x1->forward(x);
    x = torch::relu(x);
    x = conv3x32->forward(x);
    x = torch::relu(x);
    x = torch::cat({x,c},1);
    return x;
}
template<typename T>
class resnet50:public torch::nn::Module
{
    public:
        int inch,outch;
        std::unordered_map<std::string,std::vector<int>> mp;
        resblock<T> * res = new resblock<T>(mp);
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::BatchNorm bn1=nullptr;
        torch::nn::Conv2d fc=nullptr;
    public:
        resnet50(int ch_in,int ch_out,std::unordered_map<std::string,std::vector<int>> ch);
        T forward(T x);
};
template<typename T>
resnet50<T>::resnet50(int ch_in,int ch_out,std::unordered_map<std::string,std::vector<int>> ch):inch(ch_in),outch(ch_out),mp(ch),conv1(torch::nn::Conv2dOptions(inch,64,3).padding(1)),bn1(64),fc(torch::nn::Conv2dOptions(128,outch,3).padding(1))
{
    register_module("conv1",conv1);
    register_module("fc",fc);
    register_module("bn1",bn1);
}
template<typename T>
T resnet50<T>::forward(T x)
{
    x = torch::relu(conv1->forward(x));
    x = bn1->forward(x);
    x = res->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    return x;
}
