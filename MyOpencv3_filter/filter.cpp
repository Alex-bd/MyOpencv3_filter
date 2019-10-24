/*
���ڽ���������
��ֵ�˲���ֻ�ǰ�ͼ����ĸ���ģ����û����������
��ֵ�˲����ܹ��ܺõ�������������
��˹�˲�����������������ˮƽ�;�ֵ�˲�Ч�������ã�����ͼ��ģ���̶ȱȾ�ֵ�˲�Ҫ��һ��
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


void main() {
	Mat image = imread("D:/CODE/MyOpenCV3/MyOpenCV3/image/2.png");

	Mat salt_Image;
	image.copyTo(salt_Image);	
	salt(salt_Image, 3000);			//���ɽ�������

	Mat imagejunzhi(image.size(), image.type());
	Mat image2;
	averFiltering(salt_Image, imagejunzhi);		//��ֵ�˲�
	//blur(salt_Image, image2, Size(3, 3));//openCV���Դ��ľ�ֵ�˲�����
	imshow("ԭͼ", image);
	imshow("�Զ����ֵ�˲�", imagejunzhi);
	//imshow("openCV�Դ��ľ�ֵ�˲�", image2);
	Mat imagemedian;
	medianFilter(salt_Image,imagemedian);	//��ֵ�˲�
	imshow("�Զ�����ֵ�˲�",imagemedian);
	Mat imagegaussian;	
	myGaussianFilter(&salt_Image, &imagegaussian, 5, 1.5f);	//��˹�˲�
	imshow("�Զ����˹�˲�", imagegaussian);
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
void gaussian(cv::Mat* src, double** _array, int _size)
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
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
					}
				}
				// �����м���,�������õ�ֵ��û�м����ֵ���ܻ���
				temp.ptr<uchar>(i)[j] = sum;

			}
		}
	}

	// ����ԭͼ
	��* _src�� = temp.clone();
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