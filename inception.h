#include<torch/torch.h>
#include<iostream>
template<typename T>
class InceptionBlock:public torch::nn::Module
{
    public:
        int in_ch;
        torch::Tensor t1,t2,t3,t4;
        torch::Tensor tc;
        torch::nn::Conv2d upconv1=nullptr;
        torch::nn::Sequential seq1=nullptr;
        torch::nn::Sequential seq2=nullptr;
        torch::nn::Sequential seq3=nullptr;
        torch::nn::Sequential seq4=nullptr;
    public:
        InceptionBlock(int inch);
        T forward(T x);        
};
template<typename T>
InceptionBlock<T>::InceptionBlock(int inch):in_ch(inch),seq1(torch::nn::Conv2d(torch::nn::Conv2dOptions(inch,inch,3).padding(1)),torch::nn::Functional(torch::relu),torch::nn::Conv2d(torch::nn::Conv2dOptions(inch,(int)(inch/4),3).padding(1)),torch::nn::Functional(torch::relu)),
                                                        seq2(torch::nn::Conv2d(torch::nn::Conv2dOptions(inch,(int)(inch/4),3).padding(1)),torch::nn::Functional(torch::relu)),
                                                        seq3(torch::nn::Conv2d(torch::nn::Conv2dOptions(inch,(int)(inch/4),3).padding(1)),torch::nn::Functional(torch::relu)),
                                                        seq4(torch::nn::Conv2d(torch::nn::Conv2dOptions(inch,(int)(inch/4),3).padding(1))),upconv1(torch::nn::Conv2dOptions(inch,inch,4).padding(1).stride(2).transposed(new bool(true)))
{
    register_module("seq1",seq1);
    register_module("seq2",seq2);
    register_module("seq3",seq3);
    register_module("seq4",seq4);
    register_module("upconv1",upconv1);
}
template<typename T>
T InceptionBlock<T>::forward(T x)
{
    t1 = seq1->forward(x);
    t1 = torch::relu(t1);
    t2 = seq2->forward(x);
    t2 = torch::relu(t2);
    t3 = seq3->forward(x);
    t3 = torch::relu(t3);
    t4 = seq4->forward(x);
    t4 = torch::relu(t4); 
    tc = torch::cat({t1,t2,t3,t4},1);
    tc = torch::max_pool2d(tc,2);
    tc = upconv1->forward(tc);
    return tc;
}
template<typename T>
class inceptionNet:public torch::nn::Module
{
    public:
        std::vector<int> ch;
        int inch,outch;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d fc=nullptr;
    public:
        inceptionNet(int in_ch,int out_ch,std::vector<int> vec);
        T forward(T x);
};
template<typename T>
inceptionNet<T>::inceptionNet(int in_ch,int out_ch,std::vector<int> vec):ch(vec),inch(in_ch),outch(out_ch),
                                                    conv1(torch::nn::Conv2dOptions(in_ch,64,3).padding(1)),
                                                    conv2(torch::nn::Conv2dOptions(64,64,3).padding(1)),
                                                    fc(torch::nn::Conv2dOptions(64,out_ch,3).padding(1))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("fc",fc);
}
template<typename T>
T inceptionNet<T>::forward(T x)
{
    x = conv1->forward(x);
    x = torch::relu(x);
    x = conv2->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    /*
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    x = (new InceptionBlock<T>(ch[0]))->forward(x);
    x = torch::relu(x);
    */
    x = fc->forward(x);
    return x;
}

