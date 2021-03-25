#include "headerfiles.h"
#include "include/cmdline.h"

#define GRIDSIZEX 128
#define BLOCKSIZEX 256



int main(int argc, char** argv)
{
//  to run:
//	nvcc -arch=sm_20 presort.cu 
//	./a.out will print usage
	cudaError_t cudaerr = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 256);
	int ierr;
	cmdline::parser myparser;
	myparser.footer("\n\n a GPU accelerated 3D PET OSEM image reconsruction tool.\n");
	myparser.add<std::string>("lorfile", 'l', "image LOR filename", true, "");
	myparser.add<std::string>("normfile", 'n', "normalization LOR filename", false, "");
	myparser.add<std::string>("ctmhdfile", 'h', "ct header filename(MHD) for ac correction", false, "");
	myparser.add<std::string>("ctbinfile", 'c', "ct binary filename(BIN) for ac correction", false, "");
	myparser.add<std::string>("outputname", 'o', "output image filename", false, "imageZYX.bin");
	myparser.add<int>("bsize", 'b', "batchsize", false,128*128 );
	myparser.add<int>("niter", 'i', "number of iteration", false, 1);
	myparser.add("ac", 'a', "using attenuation correction");
	myparser.parse_check(argc, argv);

	bool use_ac = myparser.exist("ac"); //是否使用衰减修正
	int iterationCount = myparser.get<int>("niter");		//迭代次数
	int batchsize = myparser.get<int>("bsize");			//批次大小
	
	char* norm_lor_path = const_cast<char*>(myparser.get<std::string>("normfile").c_str());
	char* lor_path = const_cast<char*>(myparser.get<std::string>("lorfile").c_str());
	char* ct_mhd_path = const_cast<char*>(myparser.get<std::string>("ctmhdfile").c_str());
	char* ct_bin_path = const_cast<char*>(myparser.get<std::string>("ctbinfile").c_str());
	char* output_name = const_cast<char*>(myparser.get<std::string>("outputname").c_str());
	//TO DO
	PrintConfig();

	int totalnumoflines,i;
	int shouldNormalize = 0;//效率修正 TODO
	//if (norm_lor_path != "") shouldNormalize = 1;
	
	
	double totalDeviceMemoryUsed=0;
	float * norm_image = (float *)malloc(sizeof(float)*Nx*Ny*Nz);
	float * dev_norm_image;
	cudaMalloc ( ( void**)&dev_norm_image, Nx*Ny*Nz * sizeof(float) );
	totalDeviceMemoryUsed += (double)(4*Nx*Ny*Nz * sizeof(float));
	printf("(MEMORY): allocating normalization, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);
	int numoflinesForNorm=0;
	
	if(shouldNormalize >0)//效率修正
	{
		numoflinesForNorm=GetLines(norm_lor_path);
		printf("Calculating normalization image\n");
		CalcNormImage(norm_image, numoflinesForNorm, norm_lor_path);
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


	// get number of lines from lor files
	totalnumoflines=GetLines(lor_path);
	if( totalnumoflines <= 0)
	{
		printf("Empty lor file.\n");
		exit(1);
	}

	printf("Num of LORs is: %d\n",totalnumoflines);

	FILE * lor_data;
  	lor_data = fopen(lor_path, "r");
   	if (lor_data == NULL) {
		printf("lor data file %s not found\n", lor_path);
		exit(1);
	}
	else 
	{
		printf("lor data file %s is found\n", lor_path);
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
	nlines=batchsize;
	CUDAlor* lines;
	cudaMalloc ( ( void**)&lines, sizeof(CUDAlor) * nlines );	// 11 elements for the lines structure

	totalDeviceMemoryUsed += (double)(sizeof(CUDAlor) * nlines );
	printf("(MEMORY): allocating memory to store temp lor data for forward projection, device memory used: %lf MB\n", totalDeviceMemoryUsed/1048576.0);



	/*衰减修正初始化*/
	float* device_attenuation_matrix; //显存衰减矩阵


	short* host_ct_matrix_short = nullptr;
	short* dev_ct_matrix_short = nullptr; //CT矩阵 TO DO

	CTdims* dev_ctdim,*host_ctdim;			//CT矩阵定义
	int xdim, ydim, zdim, max_len;


	int bsizeac = nlines;//AC批次大小 此时等于重建批次大小
		//计算修正所需变量
	LineStatus* dev_linestat; //LOR stat
	float* dev_amin, * dev_amax;
	float* dev_tempvec_x_4f, * dev_tempvec_y_4f, * dev_tempvec_z_4f;
	float* dev_tempmat_alphas;
	float* dev_mat_alphas;
	float* dev_dis;
	int* dev_alphavecsize;
	VoxelID* dev_voxelidvec;
	//存储结果所需变量
	float* dev_linesxz_attvalue_list, * dev_linesyz_attvalue_list;

	//15个device 变量
	//6 个host 变量


	//申请内存 转换CT
	if (use_ac) {


		//device ct信息结构体初始化
		cudaMalloc((void**)&dev_ctdim, sizeof(CTdims));
		cudaMemset(dev_ctdim, 0, sizeof(CTdims));
		//host ct信息结构体初始化
		host_ctdim = (CTdims*)malloc(sizeof(CTdims));
		memset(host_ctdim, 0, sizeof(CTdims));

		//从mhd文件中读取CT信息
		ierr = genctdim(host_ctdim, ct_mhd_path);
		if (ierr != 0) //TO DO 判断err
		{
			exit(ierr);
		}
		

		//得到ct总voxel个数, 初始化host_attenu_matrix衰减矩阵
		size_t ctvoxcount = host_ctdim->xdim * host_ctdim->ydim * host_ctdim->zdim;
		cudaMalloc((void**)&device_attenuation_matrix, ctvoxcount * sizeof(float));
		cudaMemset(device_attenuation_matrix, 0, ctvoxcount * sizeof(float));

		//统计
		totalDeviceMemoryUsed += (double)(2 * sizeof(float) * ctvoxcount);
		printf("(MEMORY): allocating memory to store temp attenuation matrix, device memory used: %lf MB\n", totalDeviceMemoryUsed / 1048576.0);

		printf("(INFO): converting ct matrix values into attenuation values.\n");
		ierr = genacmatrix(device_attenuation_matrix, host_ctdim, ct_bin_path); //将CT矩阵转化为衰减值
		if (ierr != 0)
		{
			exit(ierr);
		}


		printf("(INFO): done.\n");
		if (DebugFile > 0) {
			SaveImageToFile(device_attenuation_matrix, "ATT_IMAGE.bin", ctvoxcount);//保存衰减矩阵到文件
		}
		cudaMemcpy(host_ctdim, dev_ctdim, sizeof(CTdims), cudaMemcpyDeviceToHost);

		xdim = host_ctdim->xdim;
		ydim = host_ctdim->ydim;
		zdim = host_ctdim->zdim;
		max_len = xdim + ydim + zdim + 3 + 2; //(dim+1) 以及每dim之间 有1余量
		free(host_ctdim);

		size_t onelinebuffersize = 0; //统计每个LOR所需的显存大小

		
		//申请显存
		cudaMalloc((void**)&dev_linestat, sizeof(LineStatus) * bsizeac);
		cudaMemset((void*)dev_linestat, 0, sizeof(LineStatus) * bsizeac);
		onelinebuffersize += sizeof(LineStatus);
		cudaMalloc((void**)&dev_tempvec_x_4f, sizeof(float) * bsizeac * 4);
		onelinebuffersize += sizeof(float);
		cudaMalloc((void**)&dev_tempvec_y_4f, sizeof(float) * bsizeac * 4);
		onelinebuffersize += sizeof(float);
		cudaMalloc((void**)&dev_tempvec_z_4f, sizeof(float) * bsizeac * 4);
		onelinebuffersize += sizeof(float);
		cudaMalloc((void**)&dev_amin, sizeof(float) * bsizeac);
		onelinebuffersize += sizeof(float);
		cudaMalloc((void**)&dev_amax, sizeof(float) * bsizeac);
		onelinebuffersize += sizeof(float);
		cudaMalloc((void**)&dev_tempmat_alphas, sizeof(float) * bsizeac * max_len);
		cudaMemset((void*)dev_tempmat_alphas, 0, sizeof(float) * bsizeac * max_len);
		onelinebuffersize += sizeof(float) * max_len;
		cudaMalloc((void**)&dev_voxelidvec, sizeof(VoxelID) * bsizeac * max_len);
		onelinebuffersize += sizeof(VoxelID) * max_len;
		cudaMalloc((void**)&dev_dis, sizeof(float) * bsizeac * max_len);
		onelinebuffersize += sizeof(float) * max_len;
		cudaMalloc((void**)&dev_alphavecsize, sizeof(int) * bsizeac);
		cudaMemset((void*)dev_alphavecsize, 0, sizeof(int) * bsizeac);
		onelinebuffersize += sizeof(int);
		cudaMalloc((void**)&dev_mat_alphas, sizeof(float) * bsizeac * max_len);
		cudaMemset((void*)dev_mat_alphas, 0, sizeof(float) * bsizeac * max_len);
		onelinebuffersize += sizeof(float) * max_len;
		cudaDeviceSynchronize();

		totalDeviceMemoryUsed += (double)(onelinebuffersize * bsizeac);
		printf("(MEMORY): allocating memory to for attenuation calculation, device memory used: %lf MB\n", totalDeviceMemoryUsed / 1048576.0);

	}
	

	
	/*衰减修正初始化结束*/


	int totalnumoflinesxz = sizen[1];
	int totalnumoflinesyz = sizen[0];

	printf("\nlor memory are prepared now running OSEM (running batches of %d lors) \n\n", nlines);

	






	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	printf("total iteration: #%d\n",iterationCount);


	int maxxzbatch, maxyzbatch;
	maxxzbatch = ceil(totalnumoflinesxz / (float)nlines);//计算批次个数
	maxyzbatch = ceil(totalnumoflinesyz / (float)nlines);
	int realnlines;


	//开始衰减修正
	if (use_ac) {
		int maxxzbatch_acv, maxyzbatch_acv;
		maxxzbatch_acv = ceil(totalnumoflinesxz / (float)bsizeac);//计算批次个数
		maxyzbatch_acv = ceil(totalnumoflinesyz / (float)bsizeac);
	
		cudaMalloc((void**)&dev_linesxz_attvalue_list, sizeof(float) * totalnumoflinesxz);
		cudaMalloc((void**)&dev_linesyz_attvalue_list, sizeof(float) * totalnumoflinesyz);

		//计算XZ线上的衰减值
		printf("***********************************************************************************\n");
		printf("Doing attenuation correction with lor hitting xz plane (lor-xz)\n");
		printf("***********************************************************************************\n");
		for (i = 0; i < maxxzbatch_acv; i++) {
			realnlines = bsizeac;
			if ((i + 1) * bsizeac > totalnumoflinesxz) {
				realnlines = totalnumoflinesxz - i * bsizeac;
				printf("(DEBUG) LAST BATCH XZ LOR SIZE=%d\n", realnlines);
			}//防止总数少于batchsize batchsize<0出现奇怪的bug
			int noffset = i * bsizeac;
			convertolorxz << <256, 512 >> > (dev_lor_data_array, dev_indexymax, lines, realnlines, noffset);//将lor_data中lor根据index_y_max存入lines
		
			//衰减修正开始
			bsizeac = realnlines;//该批次实际的LOR个数 realnlines<=bsizeac

			calc_stat << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_linestat);
			alphaextrema << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_amin, dev_amax, dev_tempvec_x_4f, dev_tempvec_y_4f);
			alphavecs << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_amin, dev_amax, dev_tempmat_alphas, dev_mat_alphas, dev_alphavecsize);
			dist_and_ID_in_voxel << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_voxelidvec, dev_dis, dev_mat_alphas, dev_alphavecsize);
			attu_inner_product << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, device_attenuation_matrix, dev_linestat, dev_voxelidvec, dev_dis, dev_alphavecsize);


			extract_attenu_value_to_list_with_offset << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_linesxz_attvalue_list, noffset);
			//清空数据
			cudaMemset((void*)dev_linestat, 0, sizeof(LineStatus) * bsizeac);
			cudaMemset((void*)dev_tempmat_alphas, 0, sizeof(float) * bsizeac * max_len);
			cudaMemset((void*)dev_alphavecsize, 0, sizeof(int) * bsizeac);
			cudaMemset((void*)dev_mat_alphas, 0, sizeof(float) * bsizeac * max_len);
			cudaDeviceSynchronize();

			//衰减修正结束

		}
		printf("attu corr for XZ lines done.\n");
		//计算YZ线上的衰减值
		printf("***********************************************************************************\n");
		printf("Doing attenuation correction with lor hitting yz plane (lor-yz)\n");
		printf("***********************************************************************************\n");
		for (i = 0; i < maxyzbatch_acv; i++) {
			realnlines = bsizeac;
			if ((i + 1) * bsizeac > totalnumoflinesyz) {
				realnlines = totalnumoflinesyz - i * bsizeac;
				printf("(DEBUG) LAST BATCH YZ LOR SIZE=%d\n", realnlines);
			}//防止总数少于batchsize batchsize<0出现奇怪的bug
			int noffset = i * bsizeac;
			convertolorxz << <256, 512 >> > (dev_lor_data_array, dev_indexxmax, lines, realnlines, noffset);//将lor_data中lor根据index_x_max存入lines

			//衰减修正开始
			bsizeac = realnlines;//该批次实际的LOR个数 realnlines<=bsizeac

			calc_stat << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_linestat);
			alphaextrema << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_amin, dev_amax, dev_tempvec_x_4f, dev_tempvec_y_4f);
			alphavecs << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_amin, dev_amax, dev_tempmat_alphas, dev_mat_alphas, dev_alphavecsize);
			dist_and_ID_in_voxel << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, dev_linestat, dev_voxelidvec, dev_dis, dev_mat_alphas, dev_alphavecsize);
			attu_inner_product << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_ctdim, device_attenuation_matrix, dev_linestat, dev_voxelidvec, dev_dis, dev_alphavecsize);


			extract_attenu_value_to_list_with_offset << <GRIDSIZEX, BLOCKSIZEX >> > (lines, bsizeac, dev_linesyz_attvalue_list, noffset);
			//清空数据
			cudaMemset((void*)dev_linestat, 0, sizeof(LineStatus) * bsizeac);
			cudaMemset((void*)dev_tempmat_alphas, 0, sizeof(float) * bsizeac * max_len);
			cudaMemset((void*)dev_alphavecsize, 0, sizeof(int) * bsizeac);
			cudaMemset((void*)dev_mat_alphas, 0, sizeof(float) * bsizeac * max_len);
			cudaDeviceSynchronize();

			//衰减修正结束

		}
		printf("attu corr for YZ lines done.\n");
	}


	//进行三维重建
	for (int iter=0;iter<iterationCount;iter++)
	{
		printf("now iteration: #%d\n", iter);
		if (DebugInfo > 0)
		{
			printf("***********************************************************************************\n");
			printf("Doing forward and backward projection for plane xz with lor hitting xz plane (lor-xz)\n");
			printf("***********************************************************************************\n");
		}

		//TO DO 
		maxxzbatch = ceil(totalnumoflinesxz / (float)nlines);
		//maxxzbatch = 1;
		for (i= 0; i< maxxzbatch; i++)
		{
			
			realnlines = nlines;
			//realnlines = 3;
			if ((i+1) * nlines > totalnumoflinesxz) {
				realnlines = totalnumoflinesxz - i * nlines;
				printf("(DEBUG) LAST BATCH XZ LOR SIZE=%d ",realnlines);
			}//防止总数少于batchsize batchsize<0出现奇怪的bug
			else {
				printf("(DEBUG) BATCH:%d XZ LOR SIZE=%d ", i, realnlines);
			}
			int noffset = i*nlines;
			
			

			if (use_ac) {
				
				convertolorxz_ac << <256, 512 >> > (dev_lor_data_array, dev_indexymax, lines, dev_linesxz_attvalue_list, realnlines, noffset);//将lor_data中lor根据index_y存入lines
				Forwardprojxz << <256, 512 >> > (dev_image, lines, realnlines);
				Backprojxz_ac << <256, 512 >> > (dev_image, dev_back_image, lines, realnlines, 0);//changed 			
			}					//Backprojxz<<<128,128>>>(dev_image,dev_back_image,lines,realnlines,0);
			else {
				convertolorxz << <256, 512 >> > (dev_lor_data_array, dev_indexymax, lines, realnlines, noffset);//将lor_data中lor根据index_y存入lines
				Forwardprojxz << <256, 512 >> > (dev_image, lines, realnlines);
				Backprojxz << <256, 512 >> > (dev_image, dev_back_image, lines, realnlines, 0);//changed
			}
			cudaDeviceSynchronize();
			printf("Done!\n");
			
		} // if using OSEM, move the iteration to #OSEM
		
		if(DebugInfo>0)
		{
			printf("(IMAGE) rotated image 90 degrees to point to yz plane\n");
		}
		Frotate<<<256,512>>>(dev_back_image, dev_tempback_image);
		cudaDeviceSynchronize();
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
				printf("(DEBUG) LAST BATCH YZ LOR SIZE=%d ", realnlines);
			}//防止总数少于batchsize
			else {
				printf("(DEBUG) BATCH:%d YZ LOR SIZE=%d ", i, realnlines);
			}
			int noffset = i*nlines;
			

			

			if (use_ac) {

				convertoloryz_ac << <256, 512 >> > (dev_lor_data_array, dev_indexxmax, lines, dev_linesyz_attvalue_list, realnlines, noffset);
				Forwardprojyz << <256, 512 >> > (dev_image, lines, realnlines);
				Backprojyz_ac << <256, 512 >> > (dev_image, dev_back_image, lines, realnlines, 0);
			}
			else {
				convertoloryz << <256, 512 >> > (dev_lor_data_array, dev_indexxmax, lines, realnlines, noffset);
				Forwardprojyz << <256, 512 >> > (dev_image, lines, realnlines);
				Backprojyz << <256, 512 >> > (dev_image, dev_back_image, lines, realnlines, 0);
			}
			cudaDeviceSynchronize();
			printf("Done!\n");
		} // if using OSEM, move the iteration to #OSEM

		Brotate<<<256,512>>>(dev_back_image, dev_tempback_image);//转回去
		cudaMemcpy(dev_back_image, dev_tempback_image, Nx*Ny*Nz *sizeof(float ),cudaMemcpyDeviceToDevice);

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


	//SaveImageToFile(dev_image, "image.bin", Nx * Ny * Nz); //不再存为非ZYX格式
	Rrotate << <256, 512 >> > (dev_image, dev_tempback_image);//存为ZYX格式	
	SaveImageToFile(dev_tempback_image, const_cast<char *>(output_name), Nx * Ny * Nz);
	if (Nz > 2) {
		SaveImageToFile_EX(dev_tempback_image, "imageZYX_M2.bin", Nx* Ny* Nz, Nx* Ny, Nx* Ny* (Nz - 2));//去掉顶部和底部两片之后的结果
	}
	
	printf("************************************************\n");
	printf("   all done!! elapsed time is %f s\n",elapsedTime/1000.0);	
	printf("************************************************\n");

	
	cudaFree(dev_lor_data_array);	
	cudaFree(dev_image); cudaFree(dev_back_image); cudaFree(dev_tempback_image); cudaFree(lines);
	cudaFree(dev_indexxmax); cudaFree(dev_indexymax);free(sizen);//cudaFree(dev_indexzmax);



	//衰减修正相关
	if (use_ac) {
		
		cudaFree(device_attenuation_matrix);
		cudaFree(dev_ct_matrix_short);
		cudaFree(dev_linestat);
		cudaFree(dev_tempvec_x_4f);
		cudaFree(dev_tempvec_y_4f);
		cudaFree(dev_tempvec_z_4f);
		cudaFree(dev_amin);
		cudaFree(dev_amax);
		cudaFree(dev_tempmat_alphas);
		cudaFree(dev_voxelidvec);
		cudaFree(dev_dis);
		cudaFree(dev_alphavecsize);
		cudaFree(dev_mat_alphas);
		cudaFree(dev_linesxz_attvalue_list);
		cudaFree(dev_linesyz_attvalue_list);
	}

	//结束


	return 0;
}
