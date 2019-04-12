#include <torch/torch.h>
#include <iostream>
template<typename T>
class VggBlockNTwo:public torch::nn::Module
{
    public:
        int ch;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv3=nullptr;
    public:
        VggBlockNTwo(int ch_in);
        T forward(T x);
};
template<typename T>
VggBlockNTwo<T>::VggBlockNTwo(int ch_in):ch(ch_in),
                                         conv1(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1)),
                                         conv2(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1)),
                                         conv3(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("conv3",conv3);
}
template<typename T>
T VggBlockNTwo<T>::forward(T x)
{
    x = torch::relu(conv1->forward(x));
    x = conv2->forward(x);
    x = torch::relu(x);
    x = conv3->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x,2);
    return x;
}

template<typename T>
class vggF:public torch::nn::Module
{
    public:
        int ch;
        torch::nn::Conv2d conv1=nullptr;
        torch::nn::Conv2d conv2=nullptr;
        torch::nn::Conv2d conv3=nullptr;
        torch::nn::Conv2d conv4=nullptr;
    public:
        vggF(int ch_in);
        T forward(T x);
};
template<typename T>
vggF<T>::vggF(int ch_in):ch(ch_in),
                         conv1(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1)),
                         conv2(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1)),
                         conv3(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1)),
                         conv4(torch::nn::Conv2dOptions(ch_in,ch_in,3).padding(1))
{
    register_module("conv1",conv1);
    register_module("conv2",conv2);
    register_module("conv3",conv3);
    register_module("conv4",conv4);
}
template<typename T>
T vggF<T>::forward(T x)
{
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    x = conv3->forward(x);
    x = torch::relu(x);
    x = conv4->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x,2);
    return x;
}
template<typename T>
class segNet:public torch::nn::Module
{
    public:
        torch::nn::Conv2d st=nullptr;
        torch::nn::Conv2d fc=nullptr;
        torch::nn::Conv2d conv11=nullptr;
        torch::nn::Conv2d conv12=nullptr;
        torch::nn::Conv2d conv13=nullptr;
        torch::nn::Conv2d upconv1=nullptr;
        torch::nn::Conv2d conv21=nullptr;
        torch::nn::Conv2d conv22=nullptr;
        torch::nn::Conv2d conv23=nullptr;
        torch::nn::Conv2d upconv2=nullptr;
        torch::nn::Conv2d conv31=nullptr;
        torch::nn::Conv2d conv32=nullptr;
        torch::nn::Conv2d conv33=nullptr;
        torch::nn::Conv2d convup3=nullptr;
        torch::nn::Conv2d conv41=nullptr;
        torch::nn::Conv2d conv42=nullptr;
        torch::nn::Conv2d conv43=nullptr;
        torch::nn::Conv2d convup4=nullptr;
        torch::nn::Conv2d conv51=nullptr;
        torch::nn::Conv2d conv52=nullptr;
        torch::nn::Conv2d conv53=nullptr;
        torch::nn::Conv2d convup5=nullptr;
        int inch,outch;
        std::vector<int> vc;
    public:
        segNet(int inch,int outch,std::vector<int> vc);
        T forward(T x);
};
template<typename T>
segNet<T>::segNet(int inch,int outch,std::vector<int> vc):inch(inch),outch(outch),vc(vc),
                                                          st(torch::nn::Conv2dOptions(inch,vc[0],3).padding(1)),
                                                          conv11(torch::nn::Conv2dOptions(vc[4],vc[4],3).padding(1)),
                                                          conv12(torch::nn::Conv2dOptions(vc[4],vc[4],3).padding(1)),
                                                          conv13(torch::nn::Conv2dOptions(vc[4],vc[4],3).padding(1)),
                      upconv1(torch::nn::Conv2dOptions(vc[4],vc[4],4).padding(1).stride(2).transposed(new bool(true))),
                                                          conv21(torch::nn::Conv2dOptions(vc[3],vc[3],3).padding(1)),
                                                          conv22(torch::nn::Conv2dOptions(vc[3],vc[3],3).padding(1)),
                                                          conv23(torch::nn::Conv2dOptions(vc[3],vc[3],3).padding(1)),
                      upconv2(torch::nn::Conv2dOptions(vc[3],vc[3],4).padding(1).stride(2).transposed(new bool(true))),
                                                          conv31(torch::nn::Conv2dOptions(vc[2],vc[2],3).padding(1)),
                                                          conv32(torch::nn::Conv2dOptions(vc[2],vc[2],3).padding(1)),
                                                          conv33(torch::nn::Conv2dOptions(vc[2],vc[2],3).padding(1)),
                      convup3(torch::nn::Conv2dOptions(vc[2],vc[2],4).padding(1).stride(2).transposed(new bool(true))),
                                                          conv41(torch::nn::Conv2dOptions(vc[1],vc[1],3).padding(1)),
                                                          conv42(torch::nn::Conv2dOptions(vc[1],vc[1],3).padding(1)),
                                                          conv43(torch::nn::Conv2dOptions(vc[1],vc[1],3).padding(1)),
                      convup4(torch::nn::Conv2dOptions(vc[1],vc[1],4).padding(1).stride(2).transposed(new bool(true))),
                                                          conv51(torch::nn::Conv2dOptions(vc[0],vc[0],3).padding(1)),
                                                          conv52(torch::nn::Conv2dOptions(vc[0],vc[0],3).padding(1)),
                                                          conv53(torch::nn::Conv2dOptions(vc[0],vc[0],3).padding(1)),
                      convup5(torch::nn::Conv2dOptions(vc[0],vc[0],4).padding(1).stride(2).transposed(new bool(true))),
                                                          fc(torch::nn::Conv2dOptions(vc[0],outch,3).padding(1))
{
    register_module("conv11",conv11);
    register_module("conv12",conv12);
    register_module("conv13",conv13);
    register_module("upconv1",upconv1);
    register_module("conv21",conv21);
    register_module("conv22",conv22);
    register_module("conv23",conv23);
    register_module("upconv2",upconv2);
    register_module("conv31",conv31);
    register_module("conv32",conv32);
    register_module("conv33",conv33);
    register_module("convup3",convup3);
    register_module("conv41",conv41);
    register_module("conv42",conv42);
    register_module("conv43",conv43);
    register_module("convup4",convup4);
    register_module("conv51",conv51);
    register_module("conv52",conv52);
    register_module("conv53",conv53);
    register_module("upconv5",convup5);
    register_module("fc",fc);
    register_module("st",st);
}
template<typename T>
T segNet<T>::forward(T x)
{
    x = torch::relu(st->forward(x));
    x = (new VggBlockNTwo<T>(vc[0]))->forward(x);
    x = torch::relu(x);
    x = (new VggBlockNTwo<T>(vc[1]))->forward(x);
    x = torch::relu(x);
    x = (new vggF<T>(vc[2]))->forward(x);
    x = torch::relu(x);
    x = (new vggF<T>(vc[3]))->forward(x);
    x = torch::relu(x);
    x = (new vggF<T>(vc[4]))->forward(x);
    x = torch::relu(x);
    x = conv11->forward(x);
    x = torch::relu(x);
    x = torch::relu(conv12->forward(x));
    x = torch::relu(conv13->forward(x));
    x = torch::relu(upconv1->forward(x));
    x = torch::relu(conv21->forward(x));
    x = torch::relu(conv22->forward(x));
    x = torch::relu(conv23->forward(x));
    x = torch::relu(upconv2->forward(x));
    x = torch::relu(conv31->forward(x));
    x = torch::relu(conv32->forward(x));
    x = torch::relu(conv33->forward(x));
    x = torch::relu(convup3->forward(x));
    x = torch::relu(conv41->forward(x));
    x = torch::relu(conv42->forward(x));
    x = torch::relu(conv43->forward(x));
    x = torch::relu(convup4->forward(x));
    x = torch::relu(conv51->forward(x));
    x = torch::relu(conv52->forward(x));
    x = torch::relu(conv53->forward(x));
    x = torch::relu(convup5->forward(x));
    x = fc->forward(x);
    return x;
}
                                                          




