#include <torch/torch.h>
#include <iostream>
template<typename T>
class resBlock:public torch::nn::Module
{
    public:
        int ch;
        torch::Tensor x1;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv3=nullptr;
        torch::nn::Conv2d skip=nullptr;
    public:
        resBlock(int ch);
        T forward(T x);
};
template<typename T>
resBlock<T>::resBlock(int ch):ch(ch),
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
T resBlock<T>::forward(T x)
{
    x1 = conv1->forward(x);
    x1 = torch::relu(x1);
    x1 = torch::relu(conv2->forward(x1));
    x1 = torch::relu(conv3->forward(x1));
    x = torch::relu(skip->forward(x));
    x = torch::cat({x,x1},1);
    x = torch::max_pool2d(x,2);
    return x;
}
template<typename T>
class dilationBlock:public torch::nn::Module
{
    public:
        int ch;
        torch::Tensor x1;
        torch::nn::Conv2d dconv1=nullptr;
        torch::nn::Conv2d dconv2=nullptr;
        torch::nn::Conv2d dconv3=nullptr;
        torch::nn::Conv2d dskip=nullptr;
    public:
        dilationBlock(int ch);
        T forward(T x);
};
template<typename T>
dilationBlock<T>::dilationBlock(int ch):ch(ch),
                                        dconv1(torch::nn::Conv2dOptions(ch,ch,3).padding(2).dilation(2)),
                                        dconv2(torch::nn::Conv2dOptions(ch,ch,3).padding(2).dilation(2)),
                                        dconv3(torch::nn::Conv2dOptions(ch,(int)(ch/2),3).padding(2).dilation(2)),
                                        dskip(torch::nn::Conv2dOptions(ch,(int)(ch/2),3).padding(2).dilation(2))
{
    register_module("dconv1",dconv1);
    register_module("dconv2",dconv2);
    register_module("dconv3",dconv3);
    register_module("dskip",dskip);
}
template<typename T>
T dilationBlock<T>::forward(T x)
{
    x1 = torch::relu(dconv1->forward(x));
    x1 = torch::relu(dconv2->forward(x));
    x1 = torch::relu(dconv3->forward(x));
    x = torch::relu(dskip->forward(x));
    x = torch::cat({x1,x},1);
    return x;
}
template<typename T>
class aspp:public torch::nn::Module
{
    public:
        int ch;
        torch::Tensor c1,c2,c3,c4;
        torch::nn::Conv2d dconv1x1=nullptr;
        torch::nn::Conv2d dconv3x6=nullptr;
        torch::nn::Conv2d dconv3x12=nullptr;
        torch::nn::Conv2d dconv3x18=nullptr;
    public:
        aspp(int ch);
        T forward(T x);
};
template<typename T>
aspp<T>::aspp(int ch):ch(ch),
                      dconv1x1(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(1).dilation(1)),
                      dconv3x6(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(6).dilation(6)),
                      dconv3x12(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(12).dilation(12)),
                      dconv3x18(torch::nn::Conv2dOptions(ch,(int)(ch/4),3).padding(18).dilation(18))
{
    register_module("dconv1x1",dconv1x1);
    register_module("dconv3x6",dconv3x6);
    register_module("dconv3x12",dconv3x12);
    register_module("dconv3x18",dconv3x18);
}
template<typename T>
T aspp<T>::forward(T x)
{
    c1 = torch::relu(dconv1x1->forward(x));
    c2 = torch::relu(dconv3x6->forward(x));
    c3 = torch::relu(dconv3x12->forward(x));
    c4 = torch::relu(dconv3x18->forward(x));
    x = torch::cat({c1,c2,c3,c4},1);
    return x;
}
template<typename T>
class deeplabV3:public torch::nn::Module
{
public:
    int inch,outch;
    std::vector<int> vec;
    torch::nn::Conv2d st=nullptr;
    torch::nn::Conv2d fc=nullptr;
    torch::nn::Conv2d dconv1=nullptr;
    torch::nn::Conv2d upconv1=nullptr;
    torch::nn::Conv2d upconv2=nullptr;
    torch::nn::Conv2d upconv3=nullptr;
public:
    deeplabV3(int inch,int outch,std::vector<int> vec);
    T forward(T x);
};
template<typename T>
deeplabV3<T>::deeplabV3(int inch,int outch,std::vector<int> vec):inch(inch),outch(outch),vec(vec),
              st(torch::nn::Conv2dOptions(inch,64,3).padding(1)),
              dconv1(torch::nn::Conv2dOptions(64,64,3).padding(2).dilation(2)),
              upconv1(torch::nn::Conv2dOptions(vec[2],vec[2],4).padding(1).stride(2).transposed(new bool(true))),
              upconv2(torch::nn::Conv2dOptions(vec[1],vec[1],4).padding(1).stride(2).transposed(new bool(true))),
              upconv3(torch::nn::Conv2dOptions(vec[0],vec[0],4).padding(1).stride(2).transposed(new bool(true))),
              fc(torch::nn::Conv2dOptions(64,outch,3).padding(1))
{
    register_module("conv1",st);
    register_module("conv2",fc);
    register_module("dconv1",dconv1);
    register_module("upconv1",upconv1);
    register_module("upconv2",upconv2);
    register_module("upconv3",upconv3);
}
template<typename T>
T deeplabV3<T>::forward(T x)
{
    x = st->forward(x);
    x = torch::relu(x);
    x = (new resBlock<T>(vec[0]))->forward(x);
    x = (new resBlock<T>(vec[1]))->forward(x);
    x = (new resBlock<T>(vec[2]))->forward(x); 
    x = (new dilationBlock<T>(vec[3]))->forward(x);
    x = (new aspp<T>(vec[4]))->forward(x);   
    x = upconv1->forward(x);
    x = torch::relu(x);
    x = upconv2->forward(x);
    x = torch::relu(x);
    x = upconv3->forward(x);
    x = torch::relu(x);
    x = fc->forward(x);
    return x;
}

