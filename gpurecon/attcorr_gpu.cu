#include"headerfiles.h"
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#define GRIDSIZEX 128
#define BLOCKSIZEX 256

//#define DEBUG_CALC_PROC
//#define DEBUG_STEP_RESULT

#define MAX_INFO_LINES 1
int batchcorr_gpu(CUDAlor* lines, int linesN, CTdims* ctdim, float* const attenuation_matrix, void* a_big_dev_buffer) {

#ifdef DEBUG_STEP_RESULT
	printf("linesN=%d\n", linesN);
#endif // DEBUG_STEP_RESULT

	char* curr_buffer_ptr = (char*) a_big_dev_buffer;
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

	//DEBUG
#ifdef DEBUG_STEP_RESULT
	FILE* fp = fopen("dumpattvalue.log", "a");
	CUDAlor* host_line = (CUDAlor*)malloc(sizeof(CUDAlor) * linesN);
	cudaMemcpy(host_line, lines, sizeof(CUDAlor) * linesN, cudaMemcpyDeviceToHost);
	CUDAlor* thisline = host_line;
	for (int i = 0; i < ((linesN< MAX_INFO_LINES)?linesN: MAX_INFO_LINES); i++) {
		thisline = host_line + i;
		fprintf(fp, "FIRST ID:%d\nx0 %f\nx1 %f\ny0 %f\ny1 %f\nz0 %f\nz1 %f\nav=%f\n", i, thisline->rx0, thisline->rx1, thisline->ry0, thisline->ry1, thisline->rz0, thisline->rz1, thisline->attcorrvalue);
	}
#endif // DEBUG_STEP_RESULT
	size_t onelinebuffersize = 0;
	cudaMalloc((void**)&linestat, sizeof(LineStatus) * linesN);
	//linestat = (LineStatus*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(LineStatus) * linesN;
	cudaMemset((void*)linestat, 0, sizeof(LineStatus) * linesN);
	onelinebuffersize += sizeof(LineStatus);
	cudaMalloc((void**)&tempvec_x_4f, sizeof(float) * linesN*4);
	//tempvec_x_4f = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN;
	onelinebuffersize += sizeof(float);
	cudaMalloc((void**)&tempvec_y_4f, sizeof(float) * linesN*4);
	//tempvec_y_4f = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN;
	onelinebuffersize += sizeof(float);
	cudaMalloc((void**)&tempvec_z_4f, sizeof(float) * linesN*4);
	//tempvec_z_4f = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN;
	onelinebuffersize += sizeof(float);
	cudaMalloc((void**)&amin, sizeof(float) * linesN);
	//amin = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN;
	onelinebuffersize += sizeof(float);
	cudaMalloc((void**)&amax, sizeof(float) * linesN);
	//amax = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN;
	onelinebuffersize += sizeof(float);
	cudaMalloc((void**)&tempmat_alphas, sizeof(float) * linesN * max_len);
	//tempmat_alphas = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN * max_len;
	cudaMemset((void*)tempmat_alphas, 0, sizeof(float) * linesN * max_len);
	onelinebuffersize += sizeof(float)* max_len;
	cudaMalloc((void**)&voxelidvec, sizeof(VoxelID) * linesN * max_len);
	//voxelidvec = (VoxelID*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(VoxelID) * linesN * max_len;
	onelinebuffersize += sizeof(VoxelID) * max_len;
	cudaMalloc((void**)&dis, sizeof(float) * linesN * max_len);
	//dis = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN * max_len;
	onelinebuffersize += sizeof(float) * max_len;
	cudaMalloc((void**)&alphavecsize, sizeof(int) * linesN);
	//alphavecsize = (int*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(int) * linesN;
	cudaMemset((void*)alphavecsize, 0, sizeof(int) * linesN);
	onelinebuffersize += sizeof(int);
	cudaMalloc((void**)&mat_alphas, sizeof(float) * linesN * max_len);
	//mat_alphas = (float*)curr_buffer_ptr;
	//curr_buffer_ptr += sizeof(float) * linesN * max_len;
	cudaMemset((void*)mat_alphas, 0, sizeof(float) * linesN * max_len);
	onelinebuffersize += sizeof(float) * max_len;
	//printf("one line buffer needed=%d",onelinebuffersize);
	cudaDeviceSynchronize();

	
	
	
	calc_stat << <GRIDSIZEX, BLOCKSIZEX >> > (lines, linesN, linestat);
	alphaextrema << <GRIDSIZEX, BLOCKSIZEX >> > (lines, linesN, ctdim, linestat, amin, amax, tempvec_x_4f, tempvec_y_4f);
	alphavecs << <GRIDSIZEX, BLOCKSIZEX >> > (lines, linesN, ctdim, linestat, amin, amax, tempmat_alphas, mat_alphas, alphavecsize);
	dist_and_ID_in_voxel << <GRIDSIZEX, BLOCKSIZEX >> > (lines, linesN, ctdim, linestat, voxelidvec,dis, mat_alphas, alphavecsize);
	attu_inner_product << <GRIDSIZEX, BLOCKSIZEX >> > (lines, linesN, ctdim, attenuation_matrix, linestat, voxelidvec, dis, alphavecsize);
	cudaDeviceSynchronize();
#ifdef DEBUG_STEP_RESULT
	cudaMemcpy(host_line, lines, sizeof(CUDAlor) * linesN, cudaMemcpyDeviceToHost);
	for (int i = 0; i < ((linesN < MAX_INFO_LINES) ? linesN : MAX_INFO_LINES); i++) {
		thisline = host_line + i;
		fprintf(fp,"END ID:%d,av=%f\n", i,  thisline->attcorrvalue);
	}
	free(host_line);
	fclose(fp);
#endif // DEBUG_STEP_RESULT

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
	cudaDeviceSynchronize();
	return 0;
}

__global__ void attu_inner_product(CUDAlor* lines, int nlines, CTdims* ctdim, float* attenuation_matrix, LineStatus* linestat, VoxelID* voxelidvec, float* distance, int* alphavecsize) {
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < nlines; line_index += blockDim.x * gridDim.x) {
#ifdef DEBUG_CALC_PROC	
		if (line_index < MAX_INFO_LINES) {
			printf("Phase Inner Product:ID:%d:stat is %d\n", line_index, linestat[line_index].done);
		}		
#endif // DEBUG_CALC_PROC
		if (linestat[line_index].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + line_index;
			int xdim = ctdim->xdim ;
			int ydim = ctdim->ydim ;
			int zdim = ctdim->zdim ;
			int max_len = xdim + ydim + zdim + 3 + 2;
			VoxelID* myVoxvec = voxelidvec + max_len * line_index;
			float* mydisvec = distance + max_len * line_index;
			float result = 0;
			int totps = alphavecsize[line_index] - 1;
			int xx, yy, zz;
			float attvoxvalue = 0;
			int myindex;
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
			printf("totalpts=%d\n", totps);
			printf("xdim=%d,ydim=%d,zdim=%d\n", xdim, ydim, zdim);
			printf("maxindex=%d\n", xdim * ydim * zdim);
			}
#endif // DEBUG_CALC_PROC
			for (int i = 0; i < totps; ++i) {
				xx = myVoxvec[i].xid;
				yy = myVoxvec[i].yid;
				zz = myVoxvec[i].zid;
				
				if (xx >= xdim || xx < 0) {
					result += 0;
				}
				else if (yy >= ydim || yy < 0) {
					result += 0;
				}
				else if (zz >= zdim || zz < 0) {
					result += 0;
				}
				else {
					myindex = zz * ydim * xdim + yy * xdim + xx;
#ifdef DEBUG_CALC_PROC
					if (line_index < MAX_INFO_LINES) {
						printf("voxelzyx(%d,%d,%d)", zz, yy, xx);
						printf("voxel %d index=%d\n", i, myindex);
					}
#endif // DEBUG_CALC_PROC
					attvoxvalue = attenuation_matrix[myindex];
					result += attvoxvalue * mydisvec[i]; //dis是实际长度in mm
#ifdef DEBUG_CALC_PROC
					if (line_index < MAX_INFO_LINES) {
						printf("voxelzyx(%d,%d,%d)=%f,dis=%f\n", zz, yy, xx, attvoxvalue, mydisvec[i]);
						printf("now result=%f\n", result);
					}
#endif // DEBUG_CALC_PROC
				}
				
			}
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
				printf("result=%f\n", result);
			}
#endif // DEBUG_CALC_PROC
			float cv = expf(-result);
			the_line->attcorrvalue = cv;
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
				printf("attvoxvalue=%f\n", cv);
			}
#endif // DEBUG_CALC_PROC
			linestat[line_index].done = true;
		}
	}__syncthreads();
}
__global__ void calc_stat(CUDAlor* lines, int nlines, LineStatus* linestat) {
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < nlines; line_index += blockDim.x * gridDim.x) {
#ifdef DEBUG_CALC_PROC	
		if (line_index < MAX_INFO_LINES) {
			printf("Phase Calc Line Ttpe:ID:%d:stat is %d\n", line_index, linestat[line_index].done);
		}
#endif // DEBUG_CALC_PROC
		if (linestat[line_index].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + line_index;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			bool calc_x = true, calc_y = true, calc_z = true;
			if (x0 - x1 == 0) {
				calc_x = false;
				
			}
			linestat[line_index].calcx = calc_x;
			if (y0 - y1 == 0)
			{
				calc_y = false;
				
			}
			linestat[line_index].calcy = calc_y;
			if (z0 - z1 == 0)
			{
				calc_z = false;
				
			}
			linestat[line_index].calcz = calc_z;
			if (!calc_x && !calc_y && !calc_z) {

				the_line->attcorrvalue = 1.0;
				linestat[line_index].done = true;
				printf("ID:%d IS A POINT.\n", line_index);
			}
			linestat[line_index].calcx = calc_x;
			linestat[line_index].calcy = calc_y;
			linestat[line_index].calcz = calc_z;
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
			printf("calc_stat:ID:%d,xok:%d,yok:%d,zok:%d\n", line_index, linestat[line_index].calcx, linestat[line_index].calcy, linestat[line_index].calcz);
			}
#endif // DEBUG_CALC_PROC
		}
	}__syncthreads();
}
__global__ void alphaextrema(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempvec_x_4, float* tempvec_y_4) {
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < nlines; line_index += blockDim.x * gridDim.x) {
#ifdef DEBUG_CALC_PROC	
		if (line_index < MAX_INFO_LINES) {
			printf("Phase Find Alpha Extrema:ID:%d:stat is %d\n", line_index, linestat[line_index].done);
		}
#endif // DEBUG_CALC_PROC
		if (linestat[line_index].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + line_index;
			float alphax0, alphay0, alphaz0;
			float alphaxn, alphayn, alphazn;
			float alphamin, alphamax;
			float* amaxvec = tempvec_x_4 + line_index * 4;//max=1+1X+1Y+1Z=4
			int amax_end = 0;
			amaxvec[0] = 1.0f;
			amax_end++;
			float* aminvec = tempvec_y_4 + line_index * 4;
			int amin_end = 0;
			aminvec[0] = 0.0f;
			amin_end++;
			bool calc_x = true, calc_y = true, calc_z = true;
			calc_x = linestat[line_index].calcx;
			calc_y = linestat[line_index].calcy;
			calc_z = linestat[line_index].calcz;
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
				amaxvec[amax_end] = fmaxf(alphax0, alphaxn);
				amax_end++;
				aminvec[amin_end] = fminf(alphax0, alphaxn);
				amin_end++;
			}
			if (calc_y) {
				alphay0 = (cty0 - y0) / (y1 - y0);
				alphayn = (ctyn - y0) / (y1 - y0);
				amaxvec[amax_end] = fmaxf(alphay0, alphayn);
				amax_end++;
				aminvec[amin_end] = fminf(alphay0, alphayn);
				amin_end++;
			}
			if (calc_z) {
				alphaz0 = (ctz0 - z0) / (z1 - z0);
				alphazn = (ctzn - z0) / (z1 - z0);
				amaxvec[amax_end] = fmaxf(alphaz0, alphazn);
				amax_end++;
				aminvec[amin_end] = fminf(alphaz0, alphazn);
				amin_end++;
			}
			alphamin = *thrust::max_element(thrust::seq,&aminvec[0], &aminvec[amin_end]);
			alphamax = *thrust::min_element(thrust::seq, &amaxvec[0], &amaxvec[amax_end]);
			amax[line_index] = alphamax;
			amin[line_index] = alphamin;
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
			printf("ID:%d,amax:%f,amin:%f\n", line_index, amax[line_index], amin[line_index]);
			}
#endif // DEBUG_CALC_PROC
			if (alphamax <= alphamin) {

				the_line->attcorrvalue = 1.0;
				linestat[line_index].done = true;
				
			}
		}
	}__syncthreads();
}

__global__ void alphavecs(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, float* amin, float* amax, float* tempmat_alphas, float* mat_alphas, int* alphavecsize)
{
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < nlines; line_index += blockDim.x * gridDim.x) {
#ifdef DEBUG_CALC_PROC	
		if (line_index < MAX_INFO_LINES) {
			printf("Phase Find Alpha Vecs:ID:%d:stat is %d\n", line_index, linestat[line_index].done);
		}
#endif // DEBUG_CALC_PROC
		if (linestat[line_index].done == false) {
			CUDAlor* the_line = (CUDAlor*)lines + line_index;
			int imin, jmin, kmin;
			int imax, jmax, kmax;
			float x0 = the_line->rx0;//lor线端点x0
			float x1 = the_line->rx1;//lor线端点x1
			float y0 = the_line->ry0;//lor线端点y0
			float y1 = the_line->ry1;//lor线端点y1
			float z0 = the_line->rz0;//lor线端点z0
			float z1 = the_line->rz1;//lor线端点z1
			float alphamax = amax[line_index];
			float alphamin = amin[line_index];
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
			calc_x = linestat[line_index].calcx;
			calc_y = linestat[line_index].calcy;
			calc_z = linestat[line_index].calcz;
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
			float* ax = tempmat_alphas + line_index * (ctnx + ctny + ctnz + 3 + 2);
			float* ay = tempmat_alphas + line_index * (ctnx + ctny + ctnz + 3 + 2) + ctnx + 1;
			float* az = tempmat_alphas + line_index * (ctnx + ctny + ctnz + 3 + 2) + ctnx + ctny + 2;
			float* result_begin_ptr = mat_alphas + line_index * (ctnx + ctny + ctnz + 3 + 2);
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
			
			//thrust::sort(thrust::seq, result_begin_ptr, result_begin_ptr + totalsize);//need low level sort


			//My sort
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
				printf("Before Sort Array Alha\n");
				for (int i = 0; i < totalsize; ++i) {
					printf("%f ", *(result_begin_ptr + i));
				}
				printf("\n");
			}
#endif // DEBUG_CALC_PROC
			int minidx;
			float temp;
			for (int i=0; i < totalsize - 1;++i) {
				minidx = i;
				for (int j = i + 1; j < totalsize; ++j) {
					if (result_begin_ptr[j] < result_begin_ptr[minidx]) {
						minidx = j;
					}
				}
				temp = result_begin_ptr[i];
				result_begin_ptr[i] = result_begin_ptr[minidx];
				result_begin_ptr[minidx] = temp;
			}

			
			//float* end_ptr = thrust::unique(thrust::seq, result_begin_ptr, result_begin_ptr + totalsize);
			//int veclen= end_ptr - result_begin_ptr;
			alphavecsize[line_index] = totalsize;
#ifdef DEBUG_CALC_PROC
			if (line_index < MAX_INFO_LINES) {
				printf("alphavecs=\n");
				for (int i = 0; i < totalsize; i++) {
					printf("%f ", *(result_begin_ptr + i));
				}
				printf("\n");
			}
#endif // DEBUG_CALC_PROC

		}
	}__syncthreads();
}
__global__ void dist_and_ID_in_voxel(CUDAlor* lines, int nlines, CTdims* ctdim, LineStatus* linestat, VoxelID* voxelidvec, float* distance,float* mat_alphas, int* alphavecsize) {
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < nlines; line_index += blockDim.x * gridDim.x) {
#ifdef DEBUG_CALC_PROC	
		if (line_index < MAX_INFO_LINES) {
			printf("Phase Calc Distance And VoxelIDs:ID:%d:stat is %d\n", line_index, linestat[line_index].done);
		}
#endif // DEBUG_CALC_PROC
		if (linestat[line_index].done == false) {
			int xdim = ctdim->xdim;
			int ydim = ctdim->ydim;
			int zdim = ctdim->zdim;
			int max_len = xdim + ydim + zdim + 3 + 2;
			float ctx0 = ctdim->x0;
			float ctxn = ctdim->x0 + ctdim->xdim * ctdim->xspacing;
			float cty0 = ctdim->y0;
			float ctyn = ctdim->y0 + ctdim->ydim * ctdim->yspacing;
			float ctz0 = ctdim->z0;
			float ctzn = ctdim->z0 + ctdim->zdim * ctdim->zspacing;
			float xspace = ctdim->xspacing;
			float yspace = ctdim->yspacing;
			float zspace = ctdim->zspacing;
			float* l_ptr = distance + max_len * line_index;
			int vec_len = alphavecsize[line_index];
			float* alphavec_ptr = mat_alphas + max_len * line_index;
			VoxelID* myvoxels = voxelidvec + max_len * line_index;
			CUDAlor* the_line = (CUDAlor*)lines + line_index;
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
				myvoxels[i].xid = mx;
				myvoxels[i].yid = my;
				myvoxels[i].zid = mz;
#ifdef DEBUG_CALC_PROC
				if (line_index < MAX_INFO_LINES) {
				printf("myvoxelxyz:%d=(%d,%d,%d)\n", i, myvoxels[i].xid, myvoxels[i].yid, myvoxels[i].zid);
				}
#endif // DEBUG_CALC_PROC
			}
		}
	}__syncthreads();
}