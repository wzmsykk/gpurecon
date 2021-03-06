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
#include "ideftype.h"

#define Nx 400
#define Ny 400
#define Nz 30
#define TOR_WIDTH 3
#define pixel_size 2.0
#define ISIGMA 0.8824
#define LogLevel 5 //5 ALL 4 Info 3 Debug 2 Warning 1 ERROR 0 NONE
#define DumpLevel 1
#define MAXFILENAME 128
//#define DEBUG




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

struct saxbc_functor
{
	float a;
	float b;
	float c;
public:
	saxbc_functor(float a_,float b_,float c_) 
	{
		a = a_;
		b = b_;
		c = c_;
	}

	__host__ __device__
		float operator()(const float& x) const
	{
		return (a * x + b)/c;
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
	float x0, x1, dx;//changed for attenuation correction
	float y0, y1, dy;
	float z0, z1, dz;//in voxel
	float rx0, rx1, ry0, ry1, rz0, rz1;//in real mm
	float value, attcorrvalue;
}CUDAlor;

typedef struct
{
	int xdim, ydim, zdim;//dim for voxels in ct
	float xspacing, yspacing, zspacing;//voxel spacing in different directions
	float x0, y0, z0;//minimum value of x,y,z planes in mm
}CTdims;
typedef struct
{
	bool calcx, calcy, calcz;
	bool done;
}LineStatus;
typedef struct
{
	int xid, yid, zid;
}VoxelID;

int batchcorr_gpu(CUDAlor* lines, int linesN, CTdims* ctdim, float* attenuation_matrix, void* a_big_dev_buffer);
__global__ void extract_attenu_value_to_list_with_offset(CUDAlor* lines, int nlines, float* extracted_value_list, int list_offset);
__global__ void calc_stat(CUDAlor* lines, int nlines, LineStatus* linestat);
__global__ void alphaextrema(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempvec_x_4, float* tempvec_y_4);
__global__ void alphavecs(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempmat_alphas, float* mat_alphas, int* alphavecsize);
__global__ void dist_and_ID_in_voxel(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, VoxelID* voxelidvec, float* distance, float* mat_alphas, int* alphavecsize);
__global__ void attu_inner_product(CUDAlor* lines, int nlines, CTdims* ctdim, float* attenuation_matrix, LineStatus* linestat, VoxelID* voxelidvec, float* distance, int* alphavecsize);

int attenucorrxyz(CUDAlor* lines, CTdims* ctdim, float* attenuation_matrix, int dbglv);
int batchcorr(CUDAlor* lines, int linesN, CTdims* ctdim, float* attenuation_matrix);




__global__ void convertolor(short *dev_lor_data_array, float *dx_array,float *dy_array,float *dz_array, int nlines);
void partlor(float *hx_array,float *hy_array,float *hz_array, int totalnumoflines, int *indexxmax, int *indexymax, int *indexzmax, int *sizen);
__global__ void convertoloryz(short *dev_lor_data_array, int *dev_indexxmax, CUDAlor*lines, int nlines, int noffset);
__global__ void convertoloryz_ac(short* dev_lor_data_array, int* dev_indexxmax, CUDAlor* lines, float* linesyz_attvalue_list, int nlines, int noffset);

__global__ void convertolorxz(short *dev_lor_data_array, int *dev_indexymax, CUDAlor*lines, int nlines, int noffset);
__global__ void convertolorxz_ac(short* dev_lor_data_array, int* dev_indexymax, CUDAlor* lines, float* linesxz_attvalue_list, int nlines, int noffset);

__global__ void Forwardprojyz( float *dev_image, CUDAlor* lines, int linesN );
__global__ void Forwardprojxz( float *dev_image, CUDAlor* lines, int linesN );
__global__ void Backprojxz( float *dev_image, float *back_image, CUDAlor* lines, int linesN ,int backProjOnly );
__global__ void Backprojyz( float *dev_image, float *back_image, CUDAlor* lines, int linesN ,int backProjOnly );
__global__ void Backprojxz_ac(float* dev_image, float* back_image, CUDAlor* lines, int linesN, int backProjOnly);
__global__ void Backprojyz_ac(float* dev_image, float* back_image, CUDAlor* lines, int linesN, int backProjOnly);
__global__ void Frotate(float* dst_image, float* src_image);
__global__ void Brotate(float* dst_image, float* src_image);
__global__ void Rrotate(float* dst_image, float* src_image);
__global__ void Fnorm(float *dev_image, float *back_image, float *dev_norm_image);


//__global__ void attenucorryz(float* lines, int linesN, CTdims* ctdim, float* attenuation_matrix);
int genctdim(CTdims* host_ctdim, char* ctheader);
int genacmatrix(float* dev_attenuation_matrix, CTdims* host_ctdim, char* ct_bin_path, bool readCT);

__global__ void genacvalue(float* attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix);
__global__ void genacvalue_fill(float* attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix, float value);
int GetLines(char* filename);
void PrintConfig();
void CalcNormImage(float *norm_image, int numoflinesForNorm, const char* filename);
__device__ Lorposition CalcLorPositionFull(short rsectorID1, short rsectorID2, short moduleID1, short moduleID2, short crystalID1, short crystalID2);
//__device__ Lorposition CalcLorPosition(short moduleID1, short moduleID2, short crystalID1, short crystalID2);
int SaveImageToFile(float * dev_image, char* filename, int size);
int SaveImageToFile_EX(float* dev_image, char* filename, int imagesize, int offset, int savesize);
int dumpAcValueAndLOR(char* filename, CUDAlor* lines, int dumpcount);
