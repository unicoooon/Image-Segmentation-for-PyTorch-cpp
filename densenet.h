#include<torch/torch.h>
template<typename T>
class convBlock:public torch::nn::Module
{
    public:
        int inch;
        torch::Tensor x1;
        torch::nn::Sequential seq=nullptr;
    public:
        convBlock(int in_ch);
        T forward(T x);
}; 
template<typename T>
convBlock<T>::convBlock(int in_ch):inch(in_ch),seq(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch,in_ch,3).padding(1)),
                              torch::nn::Functional(torch::relu),
                              torch::nn::BatchNorm(in_ch),
                              torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch,in_ch,3).padding(1)),
                              torch::nn::Functional(torch::relu),
                              torch::nn::BatchNorm(in_ch))
{
    register_module("seq",seq);
}
template<typename T>
T convBlock<T>::forward(T x)
{
    x1 = seq->forward(x);
    x = torch::cat({x1,x},1);
    return x;
}
template<typename T>
class denseBlock:public torch::nn::Module
{
    public:
        std::vector<int> st;
    public:
        denseBlock(std::vector<int> stage);
        T forward(T x);
};
template<typename T>
denseBlock<T>::denseBlock(std::vector<int> stage)
{
    st=stage;
}
template<typename T>
T denseBlock<T>::forward(T x)
{
    for(auto i:st)
    {
        x = (new convBlock<T>(i))->forward(x);
    }
    x = torch::relu(x);
    return x;
}
template<typename T>
class denseNet:public torch::nn::Module
{
    public:
        std::vector<std::vector<int>> t;
        int in_ch,out_ch;
        torch::nn::Conv2d st=nullptr;
        torch::nn::Conv2d fc=nullptr;
        torch::nn::BatchNorm bn1=nullptr;
        torch::nn::BatchNorm bn2=nullptr;
    public:
        denseNet(int inch,int outch,std::vector<std::vector<int>> vec);
        T forward(T x);
};
template<typename T>
denseNet<T>::denseNet(int inch,int outch,std::vector<std::vector<int>> vec):in_ch(inch),out_ch(outch),t(vec),
                                          st(torch::nn::Conv2dOptions(inch,64,3).padding(1)),
                                          bn1(64),
                                          fc(torch::nn::Conv2dOptions(1024,outch,3).padding(1)),
                                          bn2(outch)
{
    register_module("st",st);
    register_module("fc",fc);
    register_module("bn1",bn1);
    register_module("bn2",bn2);
}
template<typename T>
T denseNet<T>::forward(T x)
{
    x = torch::relu(st->forward(x));
    x = bn1->forward(x);
    x = (new denseBlock<T>(t[0]))->forward(x);
    x = torch::relu(x);
    x = (new denseBlock<T>(t[1]))->forward(x);
    x = torch::relu(x);
    x = (new denseBlock<T>(t[2]))->forward(x);
    x = torch::relu(x);
    x = (new denseBlock<T>(t[3]))->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    x = bn2->forward(x);
    return x;
}

