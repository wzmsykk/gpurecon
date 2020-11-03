#include "headerfiles.h"

int main(int argc, char** argv)
{
//  to run:
//	nvcc -arch=sm_20 presort.cu 
//	./a.out will print usage

	if(argc <=1)
	{
		printf("usage: [./a.out] [imageLORfilename] [normalizationLORfilename] [number of iteration] [batch size]\n");
		printf("if no normalization is available: \n");
		printf("usage: [./a.out] [imageLORfilename] [number of iteration] [batch size]\n");
		printf("then image is not normalized: may have ring gaps: \n");
		exit(1);
	}

	PrintConfig();

	int totalnumoflines,i;
	int shouldNormalize=0;
	int batchsize=128*128;
	double totalDeviceMemoryUsed=0;
	float * norm_image = (float *)malloc(sizeof(float)*Nx*Ny*Nz);
	float * dev_norm_image;
	cudaMalloc ( ( void**)&dev_norm_image, Nx*Ny*Nz * sizeof(float) );
	totalDeviceMemoryUsed += (double)(4*Nx*Ny*Nz * sizeof(float));
	printf("(MEMORY): allocating normalization, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);
	int numoflinesForNorm=0;
	int iterationCount = 1;
	if(argc>4)
	{
		numoflinesForNorm=GetLines(argv[2]);
		printf("Calculating normalization image\n");
		shouldNormalize=1;
		iterationCount = atoi(argv[3]);
		batchsize=atoi(argv[4]);
		CalcNormImage(norm_image, numoflinesForNorm, argv[2]);

		FILE * save_norm_imagey;
		save_norm_imagey = fopen ("norm_image.bin" , "w");
		if (save_norm_imagey == NULL) 
		{
			printf("can not write to image file!\n");
			exit(1);
		}
		fwrite(norm_image, sizeof(float), Nx*Ny*Nz, save_norm_imagey);
		cudaMemcpy(dev_norm_image, norm_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
		free(norm_image);
	}
	else
	{
		if(argc>3)
		{
			iterationCount = atoi(argv[2]);
			batchsize=atoi(argv[3]);
		}
	}

	// get number of lines from lor files
	totalnumoflines=GetLines(argv[1]);
	if( totalnumoflines <= 0)
	{
		exit(1);
	}

	printf("Num of LORs is: %d\n",totalnumoflines);

	FILE * lor_data;
  	lor_data = fopen(argv[1], "r");
   	if (lor_data == NULL) {
		printf("lor data file not found\n");
		exit(1);
	}
	else 
	{
		printf("lor data file found as %s\n",argv[1]);
	}

	// read data from lor file:
	short *lor_data_array= (short *)malloc(sizeof(short) * totalnumoflines * 6);
	for (i=0;i<totalnumoflines;i++)
	{
		fscanf(lor_data,"%hd\t%hd\t%hd\t%hd\t%hd\t%hd\n",
			&lor_data_array[6*i],
			&lor_data_array[6*i+1],
			&lor_data_array[6*i+2],
			&lor_data_array[6*i+3],
			&lor_data_array[6*i+4],
			&lor_data_array[6*i+5]);
	}

	// copy data from local to device
	short *dev_lor_data_array;

	totalDeviceMemoryUsed += (double)(6*totalnumoflines * sizeof(short));
	printf("(MEMORY): allocating LOR data, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);

	cudaMalloc ( ( void**)&dev_lor_data_array, 6*totalnumoflines * sizeof(short) );
	cudaMemcpy(dev_lor_data_array, lor_data_array, 6*totalnumoflines *sizeof(short ),cudaMemcpyHostToDevice);
	free(lor_data_array);
	
	float * dx_array; float * dy_array; float * dz_array;
	cudaMalloc ( ( void**)&dx_array,totalnumoflines*sizeof(float));
	cudaMalloc ( ( void**)&dy_array,totalnumoflines*sizeof(float));
	cudaMalloc ( ( void**)&dz_array,totalnumoflines*sizeof(float));

	totalDeviceMemoryUsed += (double)(3*totalnumoflines * sizeof(float));
	printf("(MEMORY): allocating delta x, y, z data, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);
	printf("sorting delta x, delta y, delta z\n");
	convertolor<<<256,512>>>(dev_lor_data_array,dx_array,dy_array,dz_array,totalnumoflines);

	float *hx_array= (float *)malloc(sizeof(float)*totalnumoflines);
	float *hy_array= (float *)malloc(sizeof(float)*totalnumoflines);
	float *hz_array= (float *)malloc(sizeof(float)*totalnumoflines);	
	cudaMemcpy(hx_array, dx_array, sizeof(float)*totalnumoflines,cudaMemcpyDeviceToHost);
	cudaMemcpy(hy_array, dy_array, sizeof(float)*totalnumoflines,cudaMemcpyDeviceToHost);
	cudaMemcpy(hz_array, dz_array, sizeof(float)*totalnumoflines,cudaMemcpyDeviceToHost);
	cudaFree(dx_array);cudaFree(dy_array);cudaFree(dz_array);

	totalDeviceMemoryUsed -= (double)(3*totalnumoflines * sizeof(float));
	printf("(MEMORY): de-allocating delta x, y, z data, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);

	int *indexxmax = (int *)malloc(sizeof(int)*totalnumoflines);
	int *indexymax = (int *)malloc(sizeof(int)*totalnumoflines);
	int *indexzmax = (int *)malloc(sizeof(int)*totalnumoflines);
	int *sizen = (int *)malloc(sizeof(int)*3);
	
	partlor(hx_array,hy_array,hz_array, totalnumoflines, indexxmax, indexymax, indexzmax, sizen);	
	free(hx_array);   free(hy_array);	free(hz_array);

	int *dev_indexxmax;	int *dev_indexymax;//	int *dev_indexzmax; 	
	cudaMalloc ( ( void**)&dev_indexxmax, sizen[0] * sizeof(int) );
	cudaMemcpy(dev_indexxmax, indexxmax, sizen[0] * sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc ( ( void**)&dev_indexymax, sizen[1] * sizeof(int) );
	cudaMemcpy(dev_indexymax, indexymax, sizen[1] * sizeof(int),cudaMemcpyHostToDevice);
	free(indexxmax);   free(indexymax);   free(indexzmax);   	

	totalDeviceMemoryUsed += (double)(sizen[0] * sizeof(int));
	totalDeviceMemoryUsed += (double)(sizen[1] * sizeof(int));
	printf("(MEMORY): allocating xz, yz plane max value, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);

	float *image = (float *)malloc(sizeof(float)*Nx*Ny*Nz);
	for (i=0;i<Nx*Ny*Nz;i++){*(image+i) = 1.0;}
	float * dev_image;
	cudaMalloc ( ( void**)&dev_image, Nx*Ny*Nz * sizeof(float) );

	totalDeviceMemoryUsed += (double)(sizeof(float)*Nx*Ny*Nz);
	printf("(MEMORY): allocating output image, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);
	cudaMemcpy(dev_image, image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
	free(image);
	
	float * host_back_image = (float *)malloc(sizeof(float)*Nx*Ny*Nz);
	for (i=0;i<Nx*Ny*Nz;i++){*(host_back_image+i) = 0;}
	float * dev_back_image;
	cudaMalloc ( ( void**)&dev_back_image, Nx*Ny*Nz * sizeof(float) );

	float * dev_tempback_image;
	cudaMalloc ( ( void**)&dev_tempback_image, Nx*Ny*Nz * sizeof(float) );

	totalDeviceMemoryUsed += (double)(2*sizeof(float)*Nx*Ny*Nz);
	printf("(MEMORY): allocating temp image for back projection, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);
	cudaMemcpy(dev_tempback_image, host_back_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_back_image, host_back_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyHostToDevice);
	free(host_back_image);


	int nlines = 128*128; // can adjust this one to make recon faster (need more memory)
	nlines=batchsize;//in case of total events < default batchsize which caused blank image, change the batchsize to be less than total lines. 
	float * lines;
	cudaMalloc ( ( void**)&lines, sizeof(CUDAlor) * nlines );	// 11 elements for the lines structure

	totalDeviceMemoryUsed += (double)(sizeof(CUDAlor) * nlines );
	printf("(MEMORY): allocating memory to store temp lor data for forward projection, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);

	//new lines for attenuation correction
	float* device_attenuation_matrix;
	CTdims* ctdim,*host_ctdim;
	cudaMalloc((void**)&ctdim, sizeof(CTdims));
	host_ctdim = (CTdims*)malloc(sizeof(CTdims));
	cudaMemset(ctdim, 0, sizeof(CTdims));

	cudaMalloc((void**)&device_attenuation_matrix, Nx* Ny* Nz * sizeof(float));
	cudaMemset(device_attenuation_matrix,0, Nx * Ny * Nz * sizeof(float));

	totalDeviceMemoryUsed += (double)(2*sizeof(float) * Nx * Ny * Nz);
	printf("(MEMORY): allocating memory to store temp attenuation matrix, device memory used: %lf MB\n", totalDeviceMemoryUsed / 1048576.0);

	printf("(INFO): converting ct matrix values into attenuation values.\n");
	genacmatrix(device_attenuation_matrix,ctdim,nullptr);
	
	printf("(INFO): done.\n");
	if (DebugFile > 0) {
		SaveImageToFile(device_attenuation_matrix, "ATT_IMAGE.bin", Nx* Ny* Nz);
	}
	cudaMemcpy(host_ctdim, ctdim, sizeof(CTdims), cudaMemcpyDeviceToHost);
	//new lines end

	int totalnumoflinesxz = sizen[1];
	int totalnumoflinesyz = sizen[0];

	printf("\nlor memory are prepared now running OSEM (running batches of %d lors) \n\n", nlines);
	if(DebugInfo>0)
	{
		printf("***********************************************************************************\n");
		printf("Doing forward and backward projection for plane xz with lor hitting xz plane (lor-xz)\n");	
		printf("***********************************************************************************\n");
	}

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	printf("total iteration: #%d\n",iterationCount);


	//////debug output
	FILE* fp = fopen("dump.log", "w");
	int dumpmaxline = 3;
	CUDAlor* host_dumplines = (CUDAlor*)malloc(sizeof(CUDAlor) * nlines);
	CUDAlor* this_line;
	////end
	int dumplength;
	int maxxzbatch, maxyzbatch;
	for (int iter=0;iter<iterationCount;iter++)
	{
		//TO DO 
		maxxzbatch = ceil(totalnumoflinesxz / (float)nlines);
		//maxxzbatch = 1;
		for (i= 0; i< maxxzbatch; i++)
		{
			
			int realnlines = nlines;
			if ((i+1) * nlines > totalnumoflinesxz) {
				realnlines = totalnumoflinesxz - i * nlines;
				printf("(DEBUG) LAST BATCH XZ LOR SIZE=%d\n",realnlines);
			}//防止总数少于batchsize batchsize<0出现奇怪的bug
			int noffset = i*nlines;
			
			

			convertolorxz<<<256,512>>>(dev_lor_data_array,dev_indexymax,lines,realnlines,noffset);//将lor_data中lor根据indexxy存入lines
			//dump 
			dumplength = (dumpmaxline < realnlines) ? dumpmaxline : realnlines;
			fprintf(fp, "DUMP %d lines for run %d\n", dumplength, i);
			cudaMemcpy((void* )host_dumplines, (void*)lines, sizeof(CUDAlor) * dumplength, cudaMemcpyDeviceToHost);
			this_line = (CUDAlor*)host_dumplines;
			for (int j = 0; j <dumplength ; j++) {
				this_line = (CUDAlor*)host_dumplines + j;
				fprintf(fp, "lines:%d=\n", j);
				fprintf(fp, "x0=%f,y0=%f,z0=%f\n", this_line->rx0, this_line->ry0, this_line->rz0);
			}
			checkCudaErrors(cudaDeviceSynchronize());
			//dump end
			batchcorr_gpu(lines, realnlines, ctdim, device_attenuation_matrix);
			Forwardprojxz<<<256,512>>>(dev_image, lines, realnlines);
			Backprojxz_ac <<<256, 512 >>> (dev_image, dev_back_image, lines, realnlines, 0);//changed 			
																							//Backprojxz<<<128,128>>>(dev_image,dev_back_image,lines,realnlines,0);
		} // if using OSEM, move the iteration to #OSEM
	
		if(DebugInfo>0)
		{
			printf("(IMAGE) rotated image 90 degrees to point to yz plane\n");
		}
		Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
		if (DebugFile > 0)
		{
			SaveImageToFile(dev_back_image, "dev_back_img.bin", Nx * Ny * Nz);
			SaveImageToFile(dev_tempback_image, "dev_back_img_roted.bin", Nx * Ny * Nz);
		}
		
		if(DebugInfo>0)
		{
			printf("***********************************************************************************\n");
			printf("Doing forward and backward projection for plane yz with lor hitting yz plane (lor-yz)\n");	
			printf("***********************************************************************************\n");
		}
		cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);

		maxyzbatch = ceil(totalnumoflinesyz / (float)nlines);
		//maxyzbatch = 1;
		for (i = 0; i < maxyzbatch; i++)
		//for (i = 0; i < totalnumoflinesyz / nlines ; i++)
		{
			
			int realnlines = nlines;
			if ((i + 1) * nlines > totalnumoflinesyz) {
				realnlines = totalnumoflinesyz - i * nlines;
				printf("(DEBUG) LAST BATCH YZ LOR SIZE=%d\n", realnlines);
			}//防止总数少于batchsize
			int noffset = i*nlines;
			

			convertoloryz<<<128,128>>>(dev_lor_data_array,dev_indexxmax,lines,realnlines,noffset);
			//float* host_lines = (float*)malloc(realnlines * sizeof(CUDAlor));
			//cudaMemcpy(host_lines, lines, realnlines * sizeof(CUDAlor), cudaMemcpyDeviceToHost);
			//batchcorr(host_lines, realnlines, host_ctdim, device_attenuation_matrix);//new line for attenuation correction//new line for attenuation correction
			//cudaMemcpy(lines, host_lines, realnlines * sizeof(CUDAlor), cudaMemcpyHostToDevice);
			//free(host_lines);
			batchcorr_gpu(lines, realnlines, ctdim, device_attenuation_matrix);
			
			Forwardprojyz<<<256,512>>>(dev_image, lines, realnlines);
			Backprojyz_ac<<<256,512>>>(dev_image,dev_back_image,lines,realnlines,0);
			
		} // if using OSEM, move the iteration to #OSEM

		Brotate<<<256,512>>>(dev_back_image, dev_tempback_image);//转回去
		//Brotate << <128, 128 >> > (device_attenuation_matrix, temp_attenuation_matrix);//; no need of ROT
		cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);
		//cudaMemcpy(device_attenuation_matrix, temp_attenuation_matrix, Nx * Ny * Nz * sizeof(float), cudaMemcpyDeviceToDevice);

		// Frotate<<<128,128>>>(dev_back_image, dev_tempback_image);
		// cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);

		// Frotate<<<128,128>>>(dev_back_image, dev_tempback_image);
		// cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);

		if(shouldNormalize>0)
		{
			Fnorm<<<17,720>>>(dev_image,dev_back_image,dev_norm_image);
		}
		else
		{
			cudaMemcpy(dev_image, dev_back_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);
		}

		cudaMemset(dev_back_image, 0, Nx*Ny*Nz *sizeof(float ));
		cudaMemset(lines, 0, sizeof(CUDAlor) * nlines );
		// #OSEM (indicating OSEM)
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);


	SaveImageToFile(dev_image, "image.bin", Nx * Ny * Nz);
	Rrotate << <256, 512 >> > (dev_image, dev_tempback_image);//存为ZYX格式	
	SaveImageToFile(dev_tempback_image, "imageZYX.bin", Nx * Ny * Nz);
	printf("************************************************\n");
	printf("   all done!! elapsed time is %f s\n",elapsedTime/1000.0);	
	printf("************************************************\n");

	
	cudaFree(dev_lor_data_array);
	cudaFree(device_attenuation_matrix);
	cudaFree(dev_image); cudaFree(dev_back_image); cudaFree(dev_tempback_image); cudaFree(lines);
	cudaFree(dev_indexxmax); cudaFree(dev_indexymax);free(sizen);//cudaFree(dev_indexzmax);
	return 0;
}
