#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define Nx 400
#define Ny 400
#define Nz 30
#define TOR_WIDTH 3
#define pixel_size 2.0
#define ISIGMA 0.8824
#define DebugInfo 1

#define CUDAlor_size 11 //sizeofcudalorstruct

struct maxx_tuple
{
	__host__ __device__	
	int operator() (thrust::tuple<float,float,float> t)
		{
		   float dx,dy,dz;
			thrust::tie(dx,dy,dz)=t;
			if ( (dx*dx>dy*dy) && (dx*dx>dz*dz) ) 
			{	
			return 1;}
			else			
			return 0;
		}
};

struct maxy_tuple
{
	__host__ __device__	
	int operator() (thrust::tuple<float,float,float> t)
		{
		   float dx,dy,dz;
			thrust::tie(dx,dy,dz)=t;
			if ( (dy*dy>dx*dx) && (dy*dy>dz*dz) ) 
			{	
			return 1;}
			else			
			return 0;
		}
};


struct maxz_tuple
{
	__host__ __device__	
	int operator() (thrust::tuple<float,float,float> t)
		{
		   float dx,dy,dz;
			thrust::tie(dx,dy,dz)=t;
			if ( (dz*dz>dx*dx) && (dz*dz>dy*dy) ) 
			{	
			return 1;}
			else			
			return 0;
		}
};

typedef struct
{
       float x0,x1;
        float y0,y1;
        float z0,z1;
}Lorposition;

typedef struct
{
       float x0,x1,dx;//changed for attenuation correction
        float y0,y1,dy;
        float z0,z1,dz;
        float value,attcorrvalue;
}CUDAlor;

typedef struct
{
	int xdim, ydim, zdim;//dim for voxels in ct
	float xsize, ysize, zsize;//voxel sizes in different directions
	short* ctptr;
}CT;

__global__ void convertolor(short *dev_lor_data_array, float *dx_array,float *dy_array,float *dz_array, int nlines);
void partlor(float *hx_array,float *hy_array,float *hz_array, int totalnumoflines, int *indexxmax, int *indexymax, int *indexzmax, int *sizen);
__global__ void convertoloryz(short *dev_lor_data_array, int *dev_indexxmax,float *lines, int nlines, int noffset);
__global__ void convertolorxz(short *dev_lor_data_array, int *dev_indexymax,float *lines, int nlines, int noffset);
__global__ void Forwardprojyz( float *dev_image, float *lines, int linesN );
__global__ void Forwardprojxz( float *dev_image, float *lines, int linesN );
__global__ void Backprojxz( float *dev_image, float *back_image, float *lines, int linesN ,int backProjOnly );
__global__ void Backprojyz( float *dev_image, float *back_image, float *lines, int linesN ,int backProjOnly );
__global__ void Backprojxz_ac(float* dev_image, float* back_image, float* lines, int linesN, int backProjOnly);
__global__ void Backprojyz_ac(float* dev_image, float* back_image, float* lines, int linesN, int backProjOnly);
__global__ void Frotate(float *back_image, float *back_imagetemp);
__global__ void Brotate(float* back_imagetemp, float* back_image);
__global__ void Rrotate(float* imageYZX, float* imageZYX);
__global__ void Fnorm(float *dev_image, float *back_image, float *dev_norm_image);

__global__ void attenucorrxz(float* lines, int linesN, float* attenuation_matrix);
__global__ void attenucorryz(float* lines, int linesN, float* attenuation_matrix);
__global__ void genacmatrix(float* attenuation_matrix,short* ct_matrix);
int GetLines(char* filename);
void PrintConfig();
void CalcNormImage(float *norm_image, int numoflinesForNorm, char* filename);
__device__ Lorposition CalcLorPositionFull(short rsectorID1, short rsectorID2, short moduleID1, short moduleID2, short crystalID1, short crystalID2);
//__device__ Lorposition CalcLorPosition(short moduleID1, short moduleID2, short crystalID1, short crystalID2);
void SaveImageToFile(float * dev_image, char* filename, int size);
