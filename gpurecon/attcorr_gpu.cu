#include"headerfiles.h"
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
//#define MAX_TEMP_SIZE_PER_THREAD 

int batchcorr_gpu(float* lines, int linesN, CTdims* ctdim, float* attenuation_matrix) {

	LineStatus* linestat;
	float* amin, * amax;
	float* tempvec_x_4f, *tempvec_y_4f, *tempvec_z_4f;
	float* tempmat_alphas;
	float* mat_alphas;
	float* dis;
	int* alphavecsize;
	CTdims* host_ctdim; 
	host_ctdim = (CTdims*)malloc(sizeof(CTdims));
	cudaMemcpy(host_ctdim, ctdim, sizeof(CTdims), cudaMemcpyDeviceToHost);

	int xdim = host_ctdim->xdim;
	int ydim = host_ctdim->ydim;
	int zdim = host_ctdim->zdim;
	int max_len = xdim + ydim + zdim + 3 + 2;
	free(host_ctdim);

	VoxelID* voxelidvec;
	cudaMalloc((void**)&linestat, sizeof(LineStatus) * linesN);
	cudaMalloc((void**)&tempvec_x_4f, sizeof(float) * linesN*4);
	cudaMalloc((void**)&tempvec_y_4f, sizeof(float) * linesN*4);
	cudaMalloc((void**)&tempvec_z_4f, sizeof(float) * linesN*4);
	cudaMalloc((void**)&amin, sizeof(float) * linesN);
	cudaMalloc((void**)&amax, sizeof(float) * linesN);
	cudaMalloc((void**)&tempmat_alphas, sizeof(float) * linesN * max_len);
	cudaMalloc((void**)&voxelidvec, sizeof(VoxelID) * linesN * max_len);
	cudaMalloc((void**)&dis, sizeof(float) * linesN * max_len);
	cudaMalloc((void**)&alphavecsize, sizeof(float) * linesN);
	cudaMalloc((void**)&mat_alphas, sizeof(float) * linesN * max_len);

	cudaMemset((void*)linestat, 0, sizeof(bool) * linesN);
	cudaMemset((void*)tempmat_alphas, 0, sizeof(float) * linesN * max_len);
	cudaMemset((void*)mat_alphas, 0, sizeof(float) * linesN * max_len);
	cudaMemset((void*)alphavecsize, 0, sizeof(int) * linesN);
	calc_stat << <128, 256 >> > (lines, linesN, linestat);
	alphaextrema << <128, 256 >> > (lines, linesN, ctdim, linestat, amin, amax, tempvec_x_4f, tempvec_y_4f);	
	alphavecs << <128, 256 >> > (lines, linesN, ctdim, linestat, amin, amax, tempmat_alphas, mat_alphas, alphavecsize);	
	dist_and_ID_in_voxel << <128, 256 >> > (lines, linesN, ctdim, linestat, voxelidvec,dis, mat_alphas, alphavecsize);
	attu_inner_product << <128, 256 >> > (lines, linesN, ctdim, attenuation_matrix, linestat, voxelidvec, dis, alphavecsize);
	cudaFree(linestat);
	cudaFree(tempvec_x_4f);
	cudaFree(tempvec_y_4f);
	cudaFree(tempvec_z_4f);
	cudaFree(amin);
	cudaFree(amax);
	cudaFree(tempmat_alphas);
	cudaFree(voxelidvec);
	cudaFree(dis);
	cudaFree(alphavecsize);
	cudaFree(mat_alphas);
	return 0;
}

__global__ void attu_inner_product(float* lines, int nlines, CTdims* ctdim, float* attenuation_matrix, LineStatus* linestat, VoxelID* voxelidvec, float* distance, int* alphavecsize) {
	for (int thread = threadIdx.x + blockIdx.x * blockDim.x; thread < nlines; thread += blockDim.x * gridDim.x) {
		if (linestat[thread].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + thread;
			int xdim = ctdim->xdim ;
			int ydim = ctdim->ydim ;
			int zdim = ctdim->zdim ;
			int max_len = xdim + ydim + zdim + 3 + 2;
			VoxelID* myVoxvec = voxelidvec + max_len * thread;
			float* mydisvec = distance + max_len * thread;
			double result = 0;
			int totps = alphavecsize[thread] - 1;
			int xx, yy, zz;
			for (int i = 0; i < totps; ++i) {
				xx = myVoxvec[i].xid;
				yy = myVoxvec[i].yid;
				zz = myVoxvec[i].zid;
				if (xx >= xdim || xx < 0) {
					result += 0;
				}
				else if (yy >= xdim || yy < 0) {
					result += 0;
				}
				else if (zz >= xdim || zz < 0) {
					result += 0;
				}
				else {
					result += attenuation_matrix[zz * ydim * xdim + yy * xdim + xx] * mydisvec[i];
				}
				
			}
			float cv = exp(-result);
			the_line->attcorrvalue = cv;
			
			linestat[thread].done = true;
		}
	}__syncthreads();
}
__global__ void calc_stat(float* lines, int nlines, LineStatus* linestat) {
	for (int thread = threadIdx.x + blockIdx.x * blockDim.x; thread < nlines; thread += blockDim.x * gridDim.x) {
		if (linestat[thread].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + thread;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			bool calc_x = true, calc_y = true, calc_z = true;
			if (x0 - x1 == 0) {
				calc_x = false;
				linestat[thread].calcx = calc_x;
			}
			if (y0 - y1 == 0)
			{
				calc_y = false;
				linestat[thread].calcy = calc_y;
			}
			if (z0 - z1 == 0)
			{
				calc_z = false;
				linestat[thread].calcz = calc_z;
			}
			if (!calc_x && !calc_y && !calc_z) {

				the_line->attcorrvalue = 1.0;
				linestat[thread].done = true;
			}
		}
	}__syncthreads();
}
__global__ void alphaextrema(float* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempvec_x_4, float* tempvec_y_4) {
	for (int thread = threadIdx.x + blockIdx.x * blockDim.x; thread < nlines; thread += blockDim.x * gridDim.x) {
		if (linestat[thread].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + thread;
			float alphax0, alphay0, alphaz0;
			float alphaxn, alphayn, alphazn;
			float alphamin, alphamax;
			float* amaxvec = tempvec_x_4 + thread * 4;//max=1+1X+1Y+1Z=4
			int amax_end = 0;
			amaxvec[0] = 1.0f;
			amax_end++;
			float* aminvec = tempvec_y_4 + thread * 4;
			int amin_end = 0;
			aminvec[0] = 0.0f;
			amin_end++;
			bool calc_x = true, calc_y = true, calc_z = true;
			calc_x = linestat[thread].calcx;
			calc_y = linestat[thread].calcy;
			calc_z = linestat[thread].calcz;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			float ctx0 = ctdim->x0;
			float ctxn = ctdim->x0 + ctdim->xdim * ctdim->xspacing;
			float cty0 = ctdim->y0;
			float ctyn = ctdim->y0 + ctdim->ydim * ctdim->yspacing;
			float ctz0 = ctdim->z0;
			float ctzn = ctdim->z0 + ctdim->zdim * ctdim->zspacing;
			if (calc_x) {
				alphax0 = (ctx0 - x0) / (x1 - x0);
				alphaxn = (ctxn - x0) / (x1 - x0);
				amaxvec[amax_end] = fminf(alphax0, alphaxn);
				amax_end++;
				aminvec[amin_end] = fmaxf(alphax0, alphaxn);
				amin_end++;
			}
			if (calc_y) {
				alphay0 = (cty0 - y0) / (y1 - y0);
				alphayn = (ctyn - y0) / (y1 - y0);
				amaxvec[amax_end] = fminf(alphay0, alphayn);
				amax_end++;
				aminvec[amin_end] = fmaxf(alphay0, alphayn);
				amin_end++;
			}
			if (calc_z) {
				alphaz0 = (ctz0 - z0) / (z1 - z0);
				alphazn = (ctzn - z0) / (z1 - z0);
				amaxvec[amax_end] = fminf(alphaz0, alphazn);
				amax_end++;
				aminvec[amin_end] = fmaxf(alphaz0, alphazn);
				amin_end++;
			}
			alphamin = *thrust::max_element(thrust::seq,&aminvec[0], &aminvec[amin_end]);
			alphamax = *thrust::min_element(thrust::seq, &amaxvec[0], &amaxvec[amax_end]);
			amax[thread] = alphamax;
			amin[thread] = alphamin;
			if (alphamax <= alphamin) {

				the_line->attcorrvalue = 1.0;
				linestat[thread].done = true;
			}
		}
	}__syncthreads();
}

__global__ void alphavecs(float* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempmat_alphas, float* mat_alphas, int* alphavecsize)
{
	for (int thread = threadIdx.x + blockIdx.x * blockDim.x; thread < nlines; thread += blockDim.x * gridDim.x) {
		if (linestat[thread].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + thread;
			int imin, jmin, kmin;
			int imax, jmax, kmax;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			float alphamax = amax[thread];
			float alphamin = amin[thread];
			float ctx0 = ctdim->x0;
			float ctxn = ctdim->x0 + ctdim->xdim * ctdim->xspacing;
			float cty0 = ctdim->y0;
			float ctyn = ctdim->y0 + ctdim->ydim * ctdim->yspacing;
			float ctz0 = ctdim->z0;
			float ctzn = ctdim->z0 + ctdim->zdim * ctdim->zspacing;
			float xspace = ctdim->xspacing;
			float yspace = ctdim->yspacing;
			float zspace = ctdim->zspacing;
			int ctnx = ctdim->xdim;
			int ctny = ctdim->ydim;
			int ctnz = ctdim->zdim;
			bool calc_x = true, calc_y = true, calc_z = true;
			calc_x = linestat[thread].calcx;
			calc_y = linestat[thread].calcy;
			calc_z = linestat[thread].calcz;
			if ((x1 - x0) >= 0) {
				imin = ctnx - floor((ctxn - alphamin * (x1 - x0) - x0) / xspace);
				imax = floor((x0 + alphamax * (x1 - x0) - ctx0) / xspace);
			}
			else {
				imin = ctnx - floor((ctxn - alphamax * (x1 - x0) - x0) / xspace);
				imax = floor((x0 + alphamin * (x1 - x0) - ctx0) / xspace);
			}
			if ((y1 - y0) >= 0) {
				jmin = ctny - floor((ctyn - alphamin * (y1 - y0) - y0) / yspace);
				jmax = floor((y0 + alphamax * (y1 - y0) - cty0) / yspace);
			}
			else {
				jmin = ctny - floor((ctyn - alphamax * (y1 - y0) - y0) / yspace);
				jmax = floor((y0 + alphamin * (y1 - y0) - cty0) / yspace);
			}
			if ((z1 - z0) >= 0) {
				kmin = ctnz - floor((ctzn - alphamin * (z1 - z0) - z0) / zspace);
				kmax = floor((z0 + alphamax * (z1 - z0) - ctz0) / zspace);
			}
			else {
				kmin = ctnz - floor((ctzn - alphamax * (z1 - z0) - z0) / zspace);
				kmax = floor((z0 + alphamin * (z1 - z0) - ctz0) / zspace);
			}
			float* ax = tempmat_alphas + thread * (ctnx + ctny + ctnz + 3 + 2);
			float* ay = tempmat_alphas + thread * (ctnx + ctny + ctnz + 3 + 2) + ctnx + 1;
			float* az = tempmat_alphas + thread * (ctnx + ctny + ctnz + 3 + 2) + ctnx + ctny + 2;
			float* result_begin_ptr = mat_alphas + thread * (ctnx + ctny + ctnz + 3 + 2);
			if (calc_x) {
				if (x1 - x0 >= 0) {
					for (int i = 0; i < imax - imin + 1; i++) {
						//ax[i] = i + imin;
						ax[i] = (xspace * (i + imin) + (ctx0 - x0)) / (x1 - x0);
					}
				}
				else {
					for (int i = 0; i < imax - imin + 1; i++) {
						//ax[i] = imax - i;
						ax[i] = (xspace * (imax - i) + (ctx0 - x0)) / (x1 - x0);
					}
				}
			}
			if (calc_y) {
				if (y1 - y0 >= 0) {
					for (int j = 0; j < jmax - jmin + 1; j++) {
						//ay[j] = j + jmin;
						ay[j] = (yspace * (j + jmin) + (cty0 - y0)) / (y1 - y0);
					}
				}
				else {
					for (int j = 0; j < jmax - jmin + 1; j++) {
						//ay[j] = jmax - j;
						ay[j] = (yspace * (jmax - j) + (cty0 - y0)) / (y1 - y0);
					}
				}
			}
			if (calc_z) {
				if (z1 - z0 >= 0) {
					for (int k = 0; k < kmax - kmin + 1; k++) {
						//az[k] = k + kmin;
						az[k] = (zspace * (k + kmin) + (ctz0 - z0)) / (z1 - z0);
					}
				}
				else {
					for (int k = 0; k < kmax - kmin + 1; k++) {
						//az[k] = kmax - k;
						az[k] = (zspace * (kmax - k) + (ctz0 - z0)) / (z1 - z0);
					}
				}
			}
			int totalsize = 1;
			result_begin_ptr[0] = alphamin;
			int currend = 1;
			if (calc_x) {
				for (int i = 0; i < imax - imin + 1; ++i) {
					result_begin_ptr[i + currend] = ax[i];
				}
				currend += imax - imin + 1;
				totalsize += imax - imin + 1;
			}
			if (calc_y) {
				for (int i = 0; i < jmax - jmin + 1; ++i) {
					result_begin_ptr[i + currend] = ay[i];
				}
				currend += jmax - jmin + 1;
				totalsize += jmax - jmin + 1;
			}
			if (calc_z) {
				for (int i = 0; i < kmax - kmin + 1; ++i) {
					result_begin_ptr[i + currend] = az[i];
				}
				currend += kmax - kmin + 1;
				totalsize += kmax - kmin + 1;
			}
			result_begin_ptr[currend] = alphamax;
			totalsize += 1;
			currend += 1;
			thrust::sort(thrust::seq, result_begin_ptr, result_begin_ptr + totalsize);
			
			float* end_ptr = thrust::unique(thrust::seq, result_begin_ptr, result_begin_ptr + totalsize);
			alphavecsize[thread] = end_ptr - result_begin_ptr;
			
		}
	}__syncthreads();
}
__global__ void dist_and_ID_in_voxel(float* lines, int nlines, CTdims* ctdim, LineStatus* linestat, VoxelID* voxelidvec, float* distance,float* mat_alphas, int* alphavecsize) {
	for (int thread = threadIdx.x + blockIdx.x * blockDim.x; thread < nlines; thread += blockDim.x * gridDim.x) {
		if (linestat[thread].done == false) {
			int ctxplanes = ctdim->xdim + 1;
			int ctyplanes = ctdim->ydim + 1;
			int ctzplanes = ctdim->zdim + 1;
			float ctx0 = ctdim->x0;
			float ctxn = ctdim->x0 + ctdim->xdim * ctdim->xspacing;
			float cty0 = ctdim->y0;
			float ctyn = ctdim->y0 + ctdim->ydim * ctdim->yspacing;
			float ctz0 = ctdim->z0;
			float ctzn = ctdim->z0 + ctdim->zdim * ctdim->zspacing;
			float xspace = ctdim->xspacing;
			float yspace = ctdim->yspacing;
			float zspace = ctdim->zspacing;
			int max_len = ctxplanes + ctyplanes + ctzplanes + 2;
			float* l_ptr = distance + max_len * thread;
			int vec_len = alphavecsize[thread];
			float* alphavec_ptr = mat_alphas + max_len * thread;
			VoxelID* myvoxels = voxelidvec + max_len * thread;
			CUDAlor* the_line = (CUDAlor*)lines + thread;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			float D = sqrtf((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0));
			float mean = 0;
			int mx, my, mz;
			for (int i = 0; i < vec_len-1; i++) {
				l_ptr[i] = (alphavec_ptr[i + 1] - alphavec_ptr[i]) * D;
				mean = (alphavec_ptr[i + 1] + alphavec_ptr[i]) / 2.0f;
				mx=floor(((x1 - x0) * mean + (x0 - ctx0)) / xspace);
				my= floor(((y1 - y0) * mean + (y0 - cty0)) / yspace);
				mz= floor(((z1 - z0) * mean + (z0 - ctz0)) / zspace);
				myvoxels->xid = mx;
				myvoxels->yid = my;
				myvoxels-> zid = mz;
			}
		}
	}__syncthreads();
}