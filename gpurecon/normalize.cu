#include"headerfiles.h"
__global__ void Fnorm(float *dev_image, float *dev_back_image, float *dev_norm_image)
{
	for (int i=0;i<Nx;i++)
	{
		for (int j = threadIdx.x+blockIdx.x*blockDim.x;j<Ny*Nz;j += blockDim.x * gridDim.x)
		{
			dev_image[j+Ny*Nz*i] = dev_image[j+Ny*Nz*i]*(dev_back_image[j+Ny*Nz*i])/(dev_norm_image[j+Ny*Nz*i]);
		}
	}
}

void CalcNormImage(float *norm_image, int numoflinesForNorm, char* filename)
{

	for (int i=0;i<Nx*Ny*Nz;i++){*(norm_image+i) = 0;}
	float * dev_back_image;
	cudaMalloc ( ( void**)&dev_back_image, Nx*Ny*Nz * sizeof(float) );
	cudaMemcpy(dev_back_image, norm_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);


	FILE * lor_data;
	lor_data = fopen(filename, "r");
	 if (lor_data == NULL) {
	  printf("lor data file not found\n");
	  exit(1);
	}
	else 
	{
		printf("lor data file found as %s\n",filename);
	}
	
	// read data from lor file:
	short *lor_data_array= (short *)malloc(sizeof(short) * numoflinesForNorm * 4);
	for (int i=0;i<numoflinesForNorm;i++)
	{
		fscanf(lor_data,"%hd\t%hd\t%hd\t%hd\n",
			&lor_data_array[4*i],
			&lor_data_array[4*i+1],
			&lor_data_array[4*i+2],
			&lor_data_array[4*i+3]);
	}

	// copy data from local to device
	short *dev_lor_data_array;


	cudaMalloc ( ( void**)&dev_lor_data_array, 4*numoflinesForNorm * sizeof(short) );
	cudaMemcpy(dev_lor_data_array, lor_data_array, 4*numoflinesForNorm *sizeof(short ),cudaMemcpyHostToDevice);
	free(lor_data_array);
	
	float * dx_array; float * dy_array; float * dz_array;
	cudaMalloc ( ( void**)&dx_array,numoflinesForNorm*sizeof(float));
	cudaMalloc ( ( void**)&dy_array,numoflinesForNorm*sizeof(float));
	cudaMalloc ( ( void**)&dz_array,numoflinesForNorm*sizeof(float));

	convertolor<<<512,512>>>(dev_lor_data_array,dx_array,dy_array,dz_array,numoflinesForNorm);

	float *hx_array= (float *)malloc(sizeof(float)*numoflinesForNorm);
	float *hy_array= (float *)malloc(sizeof(float)*numoflinesForNorm);
	float *hz_array= (float *)malloc(sizeof(float)*numoflinesForNorm);	
	cudaMemcpy(hx_array, dx_array, sizeof(float)*numoflinesForNorm,cudaMemcpyDeviceToHost);
	cudaMemcpy(hy_array, dy_array, sizeof(float)*numoflinesForNorm,cudaMemcpyDeviceToHost);
	cudaMemcpy(hz_array, dz_array, sizeof(float)*numoflinesForNorm,cudaMemcpyDeviceToHost);
	cudaFree(dx_array);cudaFree(dy_array);cudaFree(dz_array);


	int *indexxmax = (int *)malloc(sizeof(int)*numoflinesForNorm);
	int *indexymax = (int *)malloc(sizeof(int)*numoflinesForNorm);
	int *indexzmax = (int *)malloc(sizeof(int)*numoflinesForNorm);
	int *sizen = (int *)malloc(sizeof(int)*3);
	
	partlor(hx_array,hy_array,hz_array, numoflinesForNorm, indexxmax, indexymax, indexzmax, sizen);	
	free(hx_array);   free(hy_array);	free(hz_array);

	int *dev_indexxmax;	int *dev_indexymax;//	int *dev_indexzmax; 	
	cudaMalloc ( ( void**)&dev_indexxmax, sizen[0] * sizeof(int) );
	cudaMemcpy(dev_indexxmax, indexxmax, sizen[0] * sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc ( ( void**)&dev_indexymax, sizen[1] * sizeof(int) );
	cudaMemcpy(dev_indexymax, indexymax, sizen[1] * sizeof(int),cudaMemcpyHostToDevice);
	free(indexxmax);   free(indexymax);   free(indexzmax);   	

	float *image = (float *)malloc(sizeof(float)*Nx*Ny*Nz);
	for (int i=0;i<Nx*Ny*Nz;i++){*(image+i) = 1.0;}
	float * dev_image;
	cudaMalloc ( ( void**)&dev_image, Nx*Ny*Nz * sizeof(float) );

	cudaMemcpy(dev_image, image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
	free(image);
	

	float * dev_tempback_image;
	cudaMalloc ( ( void**)&dev_tempback_image, Nx*Ny*Nz * sizeof(float) );

	cudaMemcpy(dev_tempback_image, norm_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_back_image, norm_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);


	int nlines = 256*512;
	float * lines;
	cudaMalloc ( ( void**)&lines, 7 * nlines * sizeof(float) );	// 7 elements for the lines structure


	int totalnumoflinesxz = sizen[1];
	int totalnumoflinesyz = sizen[0];


	for (int i=0; i<totalnumoflinesxz/nlines; i++)
	{
		int realnlines = nlines;
		int noffset = i*nlines;
		convertolorxz<<<256,512>>>(dev_lor_data_array,dev_indexymax,lines,realnlines,noffset);
		Forwardprojxz<<<256,512>>>(dev_image, lines, realnlines);
		Backprojxz<<<256,512>>>(dev_image,dev_back_image,lines,realnlines,1);
	}

	Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
	cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);
	for (int i=0; i<totalnumoflinesyz/nlines; i++)
	{
		int realnlines = nlines;
		int noffset = i*nlines;
		convertoloryz<<<256,512>>>(dev_lor_data_array,dev_indexxmax,lines,realnlines,noffset);
		Forwardprojyz<<<256,512>>>(dev_image, lines, realnlines);
		Backprojyz<<<256,512>>>(dev_image,dev_back_image,lines,realnlines,1);
	}

	Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
	cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);
	// Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
	// cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);
	// Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
	// cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);

	cudaMemcpy(norm_image, dev_back_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToHost);

	cudaFree(dev_lor_data_array);
	cudaFree(dev_image); cudaFree(dev_back_image); cudaFree(dev_tempback_image); cudaFree(lines);
	cudaFree(dev_indexxmax); cudaFree(dev_indexymax);free(sizen);//cudaFree(dev_indexzmax);

}