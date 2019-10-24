/*
对于椒盐噪声，
均值滤波：只是把图像处理的更加模糊，没有有效消除噪声
中值滤波：能够很好的消除椒盐噪声
高斯滤波：消除椒盐噪声的效果不好，但是图像模糊程度比均值滤波要好一点
双边滤波：
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <ctime>

using namespace std;
using namespace cv;

void averFiltering(const Mat& src, Mat& dst);	//均值滤波
void salt(Mat& image, int num);					//图像椒盐化
uchar median(uchar n1,uchar n2,uchar n3,uchar n4,uchar n5,uchar u6,uchar n7,uchar n8,uchar n9);	//求九个数的中值
void medianFilter(const Mat& src,Mat& dst);	//中值滤波

double** getGaussianArray(int size, double sigma);	//求高斯滤波的模板
void myGaussianFilter(cv::Mat* src, cv::Mat* dst, int n, double sigma);	//分离处理各个通道，最后合并
void gaussian(cv::Mat* src, double** _array, int _size);	//通过高斯函数公式来计算高斯滤波

double** get_space_array(int size ,int channels,double sigmas);	//计算双边滤波的空间权值
double* get_color_array(int size, int channels, double sigmar);// 计算双边滤波的相似度权值
void doBialteral(Mat *src, int N, double* colorArray, double** spaceArray);// 双边 扫描计算
void myBialteralFilter(Mat *src,Mat *dst,int N,double sigmas,double sigmar);//双边滤波主函数

void main() {
	//Mat image = imread("D:/photo/zg.png");
	Mat image = imread("D:/CODE/MyOpenCV3/MyOpenCV3/image/2.png");


	Mat salt_Image;
	image.copyTo(salt_Image);	
	salt(salt_Image, 9000);			//生成椒盐噪声
	imshow("椒盐噪声", salt_Image);
	//Mat imagejunzhi(image.size(), image.type());
	//Mat image2;
	//averFiltering(salt_Image, imagejunzhi);		//均值滤波
	////blur(salt_Image, image2, Size(3, 3));//openCV库自带的均值滤波函数
	//imshow("原图", image);
	//imshow("自定义均值滤波", imagejunzhi);
	////imshow("openCV自带的均值滤波", image2);
	//Mat imagemedian;
	//medianFilter(salt_Image,imagemedian);	//中值滤波
	//imshow("自定义中值滤波",imagemedian);
	//Mat imagegaussian;	
	//myGaussianFilter(&salt_Image, &imagegaussian, 3, 1.5f);	//高斯滤波
	//imshow("自定义高斯滤波", imagegaussian);

	Mat imageshuanbian(image.size(), image.type());
	//滤波 N越大越平越模糊(2 * N + 1) sigmas空间越大越模糊sigmar相似因子
	myBialteralFilter(&salt_Image, &imageshuanbian, 5, 12.5, 50);				//双边滤波
	imshow("自定义双边滤波", imageshuanbian);	

	waitKey();
}


//均值滤波
void averFiltering(const Mat& src, Mat& dst) {
	if (!src.data) return;
	//at访问像素点
	for (int i = 1; i < src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//边缘不进行处理
				dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
					src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
					src.at<Vec3b>(i + 1, j)[0]) / 9;
				dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
					src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
					src.at<Vec3b>(i + 1, j)[1]) / 9;
				dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
					src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
					src.at<Vec3b>(i + 1, j)[2]) / 9;
			}
			else {//边缘赋值
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}
//图像椒盐化
void salt(Mat& image, int num) {
	if (!image.data) return;//防止传入空图
	int i, j;
	srand(time(NULL));
	for (int x = 0; x < num; ++x) {
		i = rand() % image.rows;
		j = rand() % image.cols;
		image.at<Vec3b>(i, j)[0] = 255;
		image.at<Vec3b>(i, j)[1] = 255;
		image.at<Vec3b>(i, j)[2] = 255;
	}
}


//求九个数的中值
uchar median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];	//定义一个数组存放九个数
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		//gap = length /2; 意味着数组被分成的组数
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);	//交换
	return arr[4];//返回中值
}


//中值滤波函数
void medianFilter(const Mat& src, Mat& dst) {
	if (!src.data)return;
	Mat matdst(src.size(), src.type());	//定义大小类型相同的输出图像
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				//求各通道的中值
				matdst.at<Vec3b>(i, j)[0] = median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
					src.at<Vec3b>(i - 1, j - 1)[0]);
				matdst.at<Vec3b>(i, j)[1] = median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
					src.at<Vec3b>(i - 1, j - 1)[1]);
				matdst.at<Vec3b>(i, j)[2] = median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
					src.at<Vec3b>(i - 1, j - 1)[2]);
			}
			else
				matdst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);	//边缘赋值
		}
	matdst.copyTo(dst);//拷贝
}


/* 获取高斯分布数组 (核大小， sigma值) */
double** getGaussianArray(int arr_size, double sigma)
{
	int i, j;
	// 初始化权值数组
	double** array = new double* [arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}
	// 高斯分布计算
	int center_i, center_j;
	center_i = center_j = arr_size / 2;		//找到中心
	double pi = 3.141592653589793;		//定义π
	double sum = 0.0f;
	// 高斯函数
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				//后面进行归一化，这部分可以不用
				//0.5f *pi*(sigma*sigma) * 
				exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
				(2.0f * sigma * sigma)));
			sum += array[i][j];
		}
	}
	// 归一化求权值
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] /= sum;
			
		}
	}
	return array;
}

/* 高斯滤波 (待处理图片, 高斯分布数组， 高斯数组大小(核大小) ) */
void gaussian(cv::Mat* _src, double** _array, int _size)
{
	cv::Mat temp = (*_src).clone();	//克隆图像
	// 扫描
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			// 忽略边缘
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2)) {
				//     找到图像输入点f(i,j),以输入点为中心与核中心对齐
				//     核心为中心参考点 卷积算子=>高斯矩阵180度转向计算
				//     x y 代表卷积核的权值坐标   i j 代表图像输入点坐标
				//     卷积算子     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f代表图像输入 g代表核
				//     带入核参考点 (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj 核参考点
				//     加权求和  注意：核的坐标以左上0,0起点
				double sum = 0.0;
		
				for (int k = 0; k < _size; k++) {
					for (int l = 0; l < _size; l++) {
						//(f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj 核参考点
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
					}
				}
				// 放入中间结果,计算所得的值与没有计算的值不能混用
				temp.ptr<uchar>(i)[j] = sum;

			}
		}
	}

	// 放入原图
	(* _src) = temp.clone();
}

//彩色图像通道分离处理，每个通道都进行高斯滤波，最后合并
void myGaussianFilter(cv::Mat* src, cv::Mat* dst, int n, double sigma)
{
	// [1] 初始化
	*dst = (*src).clone();
	// [2] 彩色图片通道分离
	std::vector<cv::Mat> channels;
	cv::split(*src, channels);
	// [3] 滤波
	// [3-1] 确定高斯正态矩阵
	double** array = getGaussianArray(n, sigma);
	// [3-2] 高斯滤波处理
	for (int i = 0; i < 3; i++) {
		gaussian(&channels[i], array, n);
	}
	// [4] 合并返回
	cv::merge(channels, *dst);
	return;
}

//计算双边滤波的空间权值
double** get_space_array(int size, int channels, double sigmas)
{
	int i, j;
	double** spaceArray = new double* [size + 1];//	多一行，最后一行的第一个数据放总值
	for (i = 0; i < size + 1; i++)
	{
		spaceArray[i] = new double[size + 1];
	}
	int center_i, center_j;
	center_i = center_j = size / 2;		//找中心点
	spaceArray[size][0] = 0.0f;   //各行第一个元素赋0
	for ( i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			spaceArray[i][j] = exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
				(2.0f * sigmas * sigmas)));
			spaceArray[size][0] += spaceArray[i][j];
		}
	}
	return spaceArray;
}

//计算双边滤波的相似度权值
/* size :模板size
   channels:通道数
   sigmar :sigmar
*/
double* get_color_array(int size, int channels, double sigmar)
{
	int n;
	double* colorArray = new double[255 * channels + 2]; //最后一位放总值
	double wr = 0.0f;
	colorArray[255 * channels + 1] = 0.0f;
	for ( n = 0; n < 255 * channels + 1; n++)
	{
		colorArray[n] = exp((-1.0f * (n * n)) / (2.0f * sigmar * sigmar));	
		colorArray[255 * channels + 1] += colorArray[n];
	}
	return colorArray;
}

//双边 扫描计算
//参数说明：
/* src:原图
	N：滤波板(核)半径N
	colorArray:一维相似度指针 ，通过get_color_array求得
	spaceArray:二维空间权值指针
*/
void doBialteral(Mat* src, int N, double* colorArray, double** spaceArray)
{
	int size = (2 * N + 1);
	Mat temp = (*src).clone();		//复制图像
	for (int i = 0; i < (*src).rows; i++)
	{
		for (int  j = 0; j < (*src).cols; j++)
		{
			//忽略边缘
			if (i > (size / 2) - 1 && j > (size / 2) - 1 && i < (*src).rows - (size / 2) && j < (*src).cols - (size / 2))
			{
				//     找到图像输入点，以输入点为中心与核中心对齐
				//     核心为中心参考点 卷积算子=>高斯矩阵180度转向计算
				//     x y 代表卷积核的权值坐标   i j 代表图像输入点坐标
				//     卷积算子     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f代表图像输入 g代表核
				//     带入核参考点 (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj 核参考点
				//     加权求和  注意：核的坐标以左上0,0起点 
				double sum[3] = { 0.0,0.0,0.0 };
				int x, y, values;
				double space_color_sum = 0.0f;
				// 注意: 公式后面的点都在核大小的范围里
				// 双边公式 g(ij) =  (f1*m1 + f2*m2 + ... + fn*mn) / (m1 + m2 + ... + mn)
				// space_color_sum = (m1 + m12 + ... + mn)
				for (int k = 0; k < size; k++) {
					for (int l = 0; l < size; l++) {
						x = i - k + (size / 2);   // 原图x  (x,y)是输入点
						y = j - l + (size / 2);   // 原图y  (i,j)是当前输出点 
						//values = f(i,j) - f(k,l)的
						values = abs((*src).at<cv::Vec3b>(i, j)[0] + (*src).at<cv::Vec3b>(i, j)[1] + (*src).at<cv::Vec3b>(i, j)[2]
							- (*src).at<cv::Vec3b>(x, y)[0] - (*src).at<cv::Vec3b>(x, y)[1] - (*src).at<cv::Vec3b>(x, y)[2]);
						//(colorArray[values] * spaceArray[k][l]) 为两个高斯函数计算出的值 w(i,j,k,l)
						space_color_sum += (colorArray[values] * spaceArray[k][l]);
					}
				}
				//计算过程
				for (int k = 0; k < size; k++)
				{
					for (int l = 0; l < size; l++)
					{
						x = i - k + (size / 2);	//原图x (x,y)是输入点
						y = j - l + (size / 2);	//原图y (i,j)是当前输出点
						values = abs((*src).at<cv::Vec3b>(i, j)[0] + (*src).at<cv::Vec3b>(i, j)[1] + (*src).at<cv::Vec3b>(i, j)[2]
							- (*src).at<cv::Vec3b>(x, y)[0] - (*src).at<cv::Vec3b>(x, y)[1] - (*src).at<cv::Vec3b>(x, y)[2]);
						for (int c = 0; c < 3; c++)
						{
							sum[c] += ((*src).at<Vec3b>(x, y)[c] * colorArray[values] * spaceArray[k][l]) / space_color_sum;
						}
					}
				}
				for (int c = 0; c < 3; c++)
				{
					temp.at<Vec3b>(i, j)[c] = sum[c];
				}
			}
		}
	}
	//放入原图
	(*src) = temp.clone();
	return;
}

//双边滤波函数

void myBialteralFilter(Mat* src, Mat* dst, int N, double sigmas, double sigmar)
{
	*dst = (*src).clone();
	int size = 2 * N + 1;
	//分别计算空间权值和相似度权值
	int channles = (*dst).channels();
	double* colorArray = NULL;
	double** spaceArray = NULL;
	colorArray = get_color_array(size, channles, sigmar);
	spaceArray = get_space_array(size, channles, sigmas);
	//滤波
	doBialteral(dst, N, colorArray, spaceArray);
	return;
}