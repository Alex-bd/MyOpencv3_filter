/*
���ڽ���������
��ֵ�˲���ֻ�ǰ�ͼ����ĸ���ģ����û����Ч��������
��ֵ�˲����ܹ��ܺõ�������������
��˹�˲�����������������Ч�����ã�����ͼ��ģ���̶ȱȾ�ֵ�˲�Ҫ��һ��
˫���˲���
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <ctime>

using namespace std;
using namespace cv;

void averFiltering(const Mat& src, Mat& dst);	//��ֵ�˲�
void salt(Mat& image, int num);					//ͼ���λ�
uchar median(uchar n1,uchar n2,uchar n3,uchar n4,uchar n5,uchar u6,uchar n7,uchar n8,uchar n9);	//��Ÿ�������ֵ
void medianFilter(const Mat& src,Mat& dst);	//��ֵ�˲�

double** getGaussianArray(int size, double sigma);	//���˹�˲���ģ��
void myGaussianFilter(cv::Mat* src, cv::Mat* dst, int n, double sigma);	//���봦�����ͨ�������ϲ�
void gaussian(cv::Mat* src, double** _array, int _size);	//ͨ����˹������ʽ�������˹�˲�

double** get_space_array(int size ,int channels,double sigmas);	//����˫���˲��Ŀռ�Ȩֵ
double* get_color_array(int size, int channels, double sigmar);// ����˫���˲������ƶ�Ȩֵ
void doBialteral(Mat *src, int N, double* colorArray, double** spaceArray);// ˫�� ɨ�����
void myBialteralFilter(Mat *src,Mat *dst,int N,double sigmas,double sigmar);//˫���˲�������

void main() {
	//Mat image = imread("D:/photo/zg.png");
	Mat image = imread("D:/CODE/MyOpenCV3/MyOpenCV3/image/2.png");


	Mat salt_Image;
	image.copyTo(salt_Image);	
	salt(salt_Image, 9000);			//���ɽ�������
	imshow("��������", salt_Image);
	//Mat imagejunzhi(image.size(), image.type());
	//Mat image2;
	//averFiltering(salt_Image, imagejunzhi);		//��ֵ�˲�
	////blur(salt_Image, image2, Size(3, 3));//openCV���Դ��ľ�ֵ�˲�����
	//imshow("ԭͼ", image);
	//imshow("�Զ����ֵ�˲�", imagejunzhi);
	////imshow("openCV�Դ��ľ�ֵ�˲�", image2);
	//Mat imagemedian;
	//medianFilter(salt_Image,imagemedian);	//��ֵ�˲�
	//imshow("�Զ�����ֵ�˲�",imagemedian);
	//Mat imagegaussian;	
	//myGaussianFilter(&salt_Image, &imagegaussian, 3, 1.5f);	//��˹�˲�
	//imshow("�Զ����˹�˲�", imagegaussian);

	Mat imageshuanbian(image.size(), image.type());
	//�˲� NԽ��ԽƽԽģ��(2 * N + 1) sigmas�ռ�Խ��Խģ��sigmar��������
	myBialteralFilter(&salt_Image, &imageshuanbian, 5, 12.5, 50);				//˫���˲�
	imshow("�Զ���˫���˲�", imageshuanbian);	

	waitKey();
}


//��ֵ�˲�
void averFiltering(const Mat& src, Mat& dst) {
	if (!src.data) return;
	//at�������ص�
	for (int i = 1; i < src.rows; ++i)
		for (int j = 1; j < src.cols; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//��Ե�����д���
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
			else {//��Ե��ֵ
				dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}
//ͼ���λ�
void salt(Mat& image, int num) {
	if (!image.data) return;//��ֹ�����ͼ
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


//��Ÿ�������ֵ
uchar median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];	//����һ�������žŸ���
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//ϣ������
		//gap = length /2; ��ζ�����鱻�ֳɵ�����
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);	//����
	return arr[4];//������ֵ
}


//��ֵ�˲�����
void medianFilter(const Mat& src, Mat& dst) {
	if (!src.data)return;
	Mat matdst(src.size(), src.type());	//�����С������ͬ�����ͼ��
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				//���ͨ������ֵ
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
				matdst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);	//��Ե��ֵ
		}
	matdst.copyTo(dst);//����
}


/* ��ȡ��˹�ֲ����� (�˴�С�� sigmaֵ) */
double** getGaussianArray(int arr_size, double sigma)
{
	int i, j;
	// ��ʼ��Ȩֵ����
	double** array = new double* [arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}
	// ��˹�ֲ�����
	int center_i, center_j;
	center_i = center_j = arr_size / 2;		//�ҵ�����
	double pi = 3.141592653589793;		//�����
	double sum = 0.0f;
	// ��˹����
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				//������й�һ�����ⲿ�ֿ��Բ���
				//0.5f *pi*(sigma*sigma) * 
				exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
				(2.0f * sigma * sigma)));
			sum += array[i][j];
		}
	}
	// ��һ����Ȩֵ
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] /= sum;
			
		}
	}
	return array;
}

/* ��˹�˲� (������ͼƬ, ��˹�ֲ����飬 ��˹�����С(�˴�С) ) */
void gaussian(cv::Mat* _src, double** _array, int _size)
{
	cv::Mat temp = (*_src).clone();	//��¡ͼ��
	// ɨ��
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			// ���Ա�Ե
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2)) {
				//     �ҵ�ͼ�������f(i,j),�������Ϊ����������Ķ���
				//     ����Ϊ���Ĳο��� �������=>��˹����180��ת�����
				//     x y �������˵�Ȩֵ����   i j ����ͼ�����������
				//     �������     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f����ͼ������ g�����
				//     ����˲ο��� (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj �˲ο���
				//     ��Ȩ���  ע�⣺�˵�����������0,0���
				double sum = 0.0;
		
				for (int k = 0; k < _size; k++) {
					for (int l = 0; l < _size; l++) {
						//(f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj �˲ο���
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
					}
				}
				// �����м���,�������õ�ֵ��û�м����ֵ���ܻ���
				temp.ptr<uchar>(i)[j] = sum;

			}
		}
	}

	// ����ԭͼ
	(* _src) = temp.clone();
}

//��ɫͼ��ͨ�����봦��ÿ��ͨ�������и�˹�˲������ϲ�
void myGaussianFilter(cv::Mat* src, cv::Mat* dst, int n, double sigma)
{
	// [1] ��ʼ��
	*dst = (*src).clone();
	// [2] ��ɫͼƬͨ������
	std::vector<cv::Mat> channels;
	cv::split(*src, channels);
	// [3] �˲�
	// [3-1] ȷ����˹��̬����
	double** array = getGaussianArray(n, sigma);
	// [3-2] ��˹�˲�����
	for (int i = 0; i < 3; i++) {
		gaussian(&channels[i], array, n);
	}
	// [4] �ϲ�����
	cv::merge(channels, *dst);
	return;
}

//����˫���˲��Ŀռ�Ȩֵ
double** get_space_array(int size, int channels, double sigmas)
{
	int i, j;
	double** spaceArray = new double* [size + 1];//	��һ�У����һ�еĵ�һ�����ݷ���ֵ
	for (i = 0; i < size + 1; i++)
	{
		spaceArray[i] = new double[size + 1];
	}
	int center_i, center_j;
	center_i = center_j = size / 2;		//�����ĵ�
	spaceArray[size][0] = 0.0f;   //���е�һ��Ԫ�ظ�0
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

//����˫���˲������ƶ�Ȩֵ
/* size :ģ��size
   channels:ͨ����
   sigmar :sigmar
*/
double* get_color_array(int size, int channels, double sigmar)
{
	int n;
	double* colorArray = new double[255 * channels + 2]; //���һλ����ֵ
	double wr = 0.0f;
	colorArray[255 * channels + 1] = 0.0f;
	for ( n = 0; n < 255 * channels + 1; n++)
	{
		colorArray[n] = exp((-1.0f * (n * n)) / (2.0f * sigmar * sigmar));	
		colorArray[255 * channels + 1] += colorArray[n];
	}
	return colorArray;
}

//˫�� ɨ�����
//����˵����
/* src:ԭͼ
	N���˲���(��)�뾶N
	colorArray:һά���ƶ�ָ�� ��ͨ��get_color_array���
	spaceArray:��ά�ռ�Ȩֵָ��
*/
void doBialteral(Mat* src, int N, double* colorArray, double** spaceArray)
{
	int size = (2 * N + 1);
	Mat temp = (*src).clone();		//����ͼ��
	for (int i = 0; i < (*src).rows; i++)
	{
		for (int  j = 0; j < (*src).cols; j++)
		{
			//���Ա�Ե
			if (i > (size / 2) - 1 && j > (size / 2) - 1 && i < (*src).rows - (size / 2) && j < (*src).cols - (size / 2))
			{
				//     �ҵ�ͼ������㣬�������Ϊ����������Ķ���
				//     ����Ϊ���Ĳο��� �������=>��˹����180��ת�����
				//     x y �������˵�Ȩֵ����   i j ����ͼ�����������
				//     �������     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f����ͼ������ g�����
				//     ����˲ο��� (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj �˲ο���
				//     ��Ȩ���  ע�⣺�˵�����������0,0��� 
				double sum[3] = { 0.0,0.0,0.0 };
				int x, y, values;
				double space_color_sum = 0.0f;
				// ע��: ��ʽ����ĵ㶼�ں˴�С�ķ�Χ��
				// ˫�߹�ʽ g(ij) =  (f1*m1 + f2*m2 + ... + fn*mn) / (m1 + m2 + ... + mn)
				// space_color_sum = (m1 + m12 + ... + mn)
				for (int k = 0; k < size; k++) {
					for (int l = 0; l < size; l++) {
						x = i - k + (size / 2);   // ԭͼx  (x,y)�������
						y = j - l + (size / 2);   // ԭͼy  (i,j)�ǵ�ǰ����� 
						//values = f(i,j) - f(k,l)��
						values = abs((*src).at<cv::Vec3b>(i, j)[0] + (*src).at<cv::Vec3b>(i, j)[1] + (*src).at<cv::Vec3b>(i, j)[2]
							- (*src).at<cv::Vec3b>(x, y)[0] - (*src).at<cv::Vec3b>(x, y)[1] - (*src).at<cv::Vec3b>(x, y)[2]);
						//(colorArray[values] * spaceArray[k][l]) Ϊ������˹�����������ֵ w(i,j,k,l)
						space_color_sum += (colorArray[values] * spaceArray[k][l]);
					}
				}
				//�������
				for (int k = 0; k < size; k++)
				{
					for (int l = 0; l < size; l++)
					{
						x = i - k + (size / 2);	//ԭͼx (x,y)�������
						y = j - l + (size / 2);	//ԭͼy (i,j)�ǵ�ǰ�����
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
	//����ԭͼ
	(*src) = temp.clone();
	return;
}

//˫���˲�����

void myBialteralFilter(Mat* src, Mat* dst, int N, double sigmas, double sigmar)
{
	*dst = (*src).clone();
	int size = 2 * N + 1;
	//�ֱ����ռ�Ȩֵ�����ƶ�Ȩֵ
	int channles = (*dst).channels();
	double* colorArray = NULL;
	double** spaceArray = NULL;
	colorArray = get_color_array(size, channles, sigmar);
	spaceArray = get_space_array(size, channles, sigmas);
	//�˲�
	doBialteral(dst, N, colorArray, spaceArray);
	return;
}