#include<iostream>
#include<stdlib.h>
#include<assert.h>
#include<opencv2/opencv.hpp>
#include <ogrsf_frmts.h>s
#include <gdal_priv.h>
#include <ogr_geometry.h
#include <string>
#include <vector>
#define Size 10
int imgH=256;
int imgW=256;
cv::Mat RandData(cv::Mat data,int height,int width)
{/*
  To get a single training Tif data;
  */
    cv::Mat tmp;
    assert(height>0);
    assert(width>0);
    tmp.create(imgH,imgW,CV_32FC(4));
    //cv::Mat tmp = cv::Mat::zeros(imgH,imgW,CV_8UC3);
    for(int i=height;i<height+imgH;i++)
    {
        for(int j=width;j<width+imgW;j++)
        {
            tmp.at<cv::Vec3b>(i-height,j-width)[0] = int(data.at<cv::Vec3b>(i,j)[0]);
            tmp.at<cv::Vec3b>(i-height,j-width)[1] = int(data.at<cv::Vec3b>(i,j)[1]);
            tmp.at<cv::Vec3b>(i-height,j-width)[2] = int(data.at<cv::Vec3b>(i,j)[2]);
            tmp.at<cv::Vec3b>(i-height,j-width)[3] = int(data.at<cv::Vec3b>(i,j)[3]);
        }
    }
    return tmp;
}
cv::Mat RandMask(cv::Mat data,int height,int width)
{/*
  To get a single training Mask data;
  */
    assert(height>0);
    assert(width>0);
    cv::Mat tmp = cv::Mat::zeros(imgH,imgW,CV_8UC1);
    for(int i=height;i<height+imgH;i++)
    {
        for(int j=width;j<width+imgW;j++)
        {
            tmp.at<uchar>(i-height,j-width)= int(data.at<uchar>(i,j));
        }
    }
    return tmp;
}
float**** mat2floatData(std::vector<std::vector<cv::Mat>> mat)
{/*
  Transform mat to numpy;
  */
    assert(Size>0);
    float ****res = new float***[Size];
    for(int t =0;t<Size;t++)
    {
        res[t] = new float**[256];
        for(int i=0;i<256;i++)
        {
            res[t][i] = new float*[256];
            for(int j=0;j<256;j++)
            {
                res[t][i][j]= new float[4];
                for(int k=0;k<4;k++)
                {
                    res[t][i][j][k]=int(mat[0][t].at<cv::Vec3b>(i,j)[k]);
                }
            }
        }
    }
    return res;
}
float**** mat2floatMask(std::vector<std::vector<cv::Mat>> mat)
{/*
  Transform mat to numpy;
  */
    float ****res = new float***[Size];
    for(int t =0;t<Size;t++)
    {
        res[t] = new float**[256];
        for(int i=0;i<256;i++)
        {
            res[t][i] = new float*[256];
            for(int j=0;j<256;j++)
            {
                res[t][i][j]=new float[1];
                for(int k=0;k<1;k++)
                {
                    res[t][i][j][k]=int(mat[1][t].at<uchar>(i,j));
                }
            }
        }
    }
    return res;
}
std::vector<std::vector<cv::Mat>>DataLoader(cv::Mat data,cv::Mat mask,int batch_size)
{/*
  load batch training data,include Tif and Mask
  */
    assert(batch_size>0);
    int imgheight = data.rows;
    int imgwidth = data.cols;
    int randheight,randwidth;
    std::vector<std::vector<cv::Mat>>res;
    std::vector<cv::Mat> sdata;
    std::vector<cv::Mat> smask;
    for(int i=0;i<batch_size;i++)
    {
        randheight=rand()%(imgheight-imgH-1);
        randwidth = rand()%(imgwidth-imgW-1);
        sdata.push_back(RandData(data,randheight,randwidth));
        smask.push_back(RandMask(mask,randheight,randwidth));
    }
    res.push_back(sdata);
    res.push_back(smask);
    return res;
}
float* GetFloatData(float ****arr)
{/*
  flatten tensor;
  */
    assert(Size>0)
    float *res = new float[Size*256*256*4];
    int tmp = 0;
    for(int i=0;i<Size;i++)
    {
        for(int j=0;j<256;j++)
        {
            for(int k=0;k<256;k++)
            {
                for(int s =0;s<4;s++)
                {
                    res[tmp++]=arr[i][j][k][s];
                }
            }
        }
    }
    return res;
}
float* GetFloatMask(float ****arr)
{/*
  flatten tensor;
  */
    assert(Size>0)
    float *res = new float[Size*256*256*1];
    int tmp = 0;
    for(int i=0;i<Size;i++)
    {
        for(int j=0;j<256;j++)
        {
            for(int k=0;k<256;k++)
            {
                for(int s =0;s<1;s++)
                {
                    res[tmp++]=arr[i][j][k][s];
                }
            }
        }
    }
    return res;
}
cv::Mat GDAL2Mat(const char* fileName)
{/*
  Transform Gdal data to opencv data;
  */
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
float *GetListData(cv::Mat data)
{/*
  flatten mat;
  */
    float *t = new float[1000*1000*4];
    int temp=0;
    for(int i=0;i<1000;i++)
    {
        for(int j=0;j<1000;j++)
        {
            for(int k=0;k<4;k++)
            {
                t[temp++]=data.at<cv::Vec3d>(i,j)[k];
            }
        }
    }
    return t;
}
