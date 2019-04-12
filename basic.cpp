#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>
#include <gdal_priv.h>
#include <ogr_geometry.h>
#include<iostream>
#include <string>
using namespace std;
using namespace cv;
cv::Mat readImg(std::string path)
{
    cv::Mat img = cv::imread(path);
    if(img.empty())
    {
        std::cout<<"picture don't exit"<<std::endl;
        exit(-1);
    }
    return img;
}
int get_channels(cv::Mat img)
{
    return img.channels();
}
cv::Mat filter2d(cv::Mat src,cv::Mat &drc,cv::Mat kernel)
{
    cv::filter2D(src,drc,src.depth(),kernel);
    return drc;
}
namespace createImg
{
    cv::Mat one_method()
    {   
        cv::Mat img(3,3,CV_8UC1,Scalar(255));
        return img;
    }
    cv::Mat two_method()
    {
        Mat z;
        z.create(100,100,CV_8UC1);
        z=Scalar(255);
        return z;
    }
}
cv::Mat addweight(cv::Mat src1,float alpha,cv::Mat src2,cv::Mat & dst)
{
    cv::addWeighted(src1,alpha,src2,1-alpha,0.0,dst);
    return dst;
}
cv::Mat lightImg(cv::Mat img)
{
    cv::Mat res =cv::Mat::zeros(img.size(),img.type());
    cv::MatIterator_<Vec3b> beg,endd;
    cv::MatIterator_<Vec3b> resb,rese;
    for(beg=img.begin<Vec3b>(),resb=res.begin<Vec3b>(),endd=img.end<Vec3b>();beg!=endd;beg++,resb++)
    {
        (*resb)[0]=cv::saturate_cast<uchar>((*beg)[0]*2+10);
        (*resb)[1]=cv::saturate_cast<uchar>((*beg)[1]*2+10);
        (*resb)[2]=cv::saturate_cast<uchar>((*beg)[2]*2+10);
    }
    return res;
}
cv::Mat dline(cv::Mat &img)
{
    cv::Point ps = Point(10,10);
    cv::Point pe = Point(100,100);
    cv::Scalar color=Scalar(255,255,255);
    cv::line(img,ps,pe,color,2,LINE_8);
    return img;
}
cv::Mat drectangle(cv::Mat &img)
{
    cv::Rect rect = Rect(10,10,100,100);
    cv::Scalar color = Scalar(255,255,255);
    cv::rectangle(img,rect,color,2,LINE_8);
    return img;
}
cv::Mat dellipse(cv::Mat &img)
{
    cv::ellipse(img,Point(50,50),cv::Size(30,20),0,0,360,cv::Scalar(0,255,0),2,LINE_8);
    return img;
}
cv::Mat dcircle(cv::Mat &img)
{
    cv::circle(img,Point(50,50),50,Scalar(0,255,0),2,LINE_8);
    return img;
}
cv::Mat dputtext(cv::Mat &img)
{
    cv::putText(img,"txt",Point(50,50),CV_FONT_HERSHEY_COMPLEX,1.0,Scalar(255,255,255),1,LINE_8);
    return img;
}
cv::Mat oblur(cv::Mat img,cv::Mat &dst)
{
    cv::blur(img,dst,Size(5,5),Point(-1,-1));
    return dst;
}
cv::Mat ogaussianblur(cv::Mat img,cv::Mat &dst)
{
    cv::GaussianBlur(img,dst,Size(7,7),1,2);
    return dst;
}
cv::Mat close_op(cv::Mat src,cv::Mat &dst)
{
    cv::Mat kernel = cv::getStructuringElement(MORPH_RECT,Size(211,211),Point(-1,-1));
    cv::morphologyEx(src,dst,CV_MOP_CLOSE,kernel);
    return dst;
}
cv::Mat GDAL2Mat(const char* fileName)
{
    GDALAllRegister();
    GDALDataset *poDataset = (GDALDataset *)GDALOpen(fileName,GA_ReadOnly);
    int Cols = poDataset->GetRasterXSize();
    int Rows = poDataset->GetRasterYSize();
    int BandSize = poDataset->GetRasterCount();
    std::vector<cv::Mat> imgMat;
    float *pafScan;
    GDALRasterBand *pBand;
    for(int i = 0;i< BandSize;i++)
    {
        pBand = poDataset->GetRasterBand(i+1);
        pafScan = new float[Cols*Rows];
        (void)pBand->RasterIO(GF_Read,0,0,Cols,Rows,pafScan,Cols,Rows,GDT_Float32,0,0);
        cv::Mat A = cv::Mat(Rows,Cols,CV_32FC1,pafScan);
        imgMat.push_back(A.clone());
        A.release();
    }
    cv::Mat img;
    img.create(Rows,Cols,CV_32FC(BandSize));
    cv::merge(imgMat,img);
    imgMat.clear();
    return img;
} 
cv::Mat testV(cv::Mat img)
{
    cv::Mat res = cv::Mat::zeros(img.size(),img.type());
    cv::MatIterator_<Vec6f> beg,endd;
    cv::MatIterator_<Vec6f> resb,rese;
    for(beg=img.begin<Vec6f>(),resb=res.begin<Vec6f>(),endd=img.end<Vec6f>();beg!=endd;beg++,resb++)
    {   
        (*resb)[0]=1220;
        (*resb)[1]=1220;
        (*resb)[2]=12200;
        (*resb)[3]=12200;
        (*resb)[4]=12200;
        (*resb)[5]=12200;
        (*resb)[6]=12200;
    }
    return res;
}
float *tensor2float(torch::Tensor x)
{
    float *res  = new float[x.size(0)*x.size(1)];
    auto iter = x.accessor<float,2>();
    int tmp = 0;
    for(int i=0;i<x.size(0);i++)
    {
        for(int j=0;j<x.size(1);j++)
        {
            res[tmp++]=iter[i][j];
        }
    }
    return res;
}
std::tuple<torch::Tensor,torch::Tensor> getUnique(torch::Tensor x)
{
    return torch::_unique(x);
}
torch::Tensor TensorOptions(torch::Tensor x)
{
    return x.to(torch::kFloat32).to(torch::kCPU);
}
torch::Tensor randTensorOptions()
{
    return torch::randn({256,256},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}
torch::Tensor Transpose(torch::Tensor x)
{
    return torch::transpose(torch::transpose(x,1,2),0,1);
}
torch::Tensor targmax(torch::Tensor x)
{
    return torch::argmax(x,2,true);
}
torch::Tensor tsqueeze(torch::Tensor x)
{
    return torch::squeeze(x);
}
torch::Tensor tunsqueeze(torch::Tensor x)
{
    return torch::unsqueeze(x,-1);
}
torch::Tensor tfrom_blob(float *arr)
{
    return torch::from_blob(arr,{3,256,256},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}
float *getFloatArray()
{
     float *res = new float[3*256*256];
     for(int i=0;i<3*256*256;i++)
     {
         res[i]=i;
     }
     return res;
}
int main()
{
    float * res = getFloatArray();
    torch::Tensor t = tfrom_blob(res);
    std::cout<<t.sizes()<<std::endl;   
    return 0;
}











