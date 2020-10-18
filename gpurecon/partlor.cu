#include "headerfiles.h"


void partlor(float *hx_array,float *hy_array,float *hz_array, int totalnumoflines, int *indexxmax, int *indexymax, int *indexzmax, int *sizen)
{
	
	std::vector<float> hx_vec(hx_array,hx_array+totalnumoflines);//使用数组指针初始化std::vector
	std::vector<float> hy_vec(hy_array,hy_array+totalnumoflines);
	std::vector<float> hz_vec(hz_array,hz_array+totalnumoflines);	

		
	thrust::device_vector<float> dx_vec = hx_vec;
	thrust::device_vector<float> dy_vec = hy_vec;
	thrust::device_vector<float> dz_vec = hz_vec;
	thrust::device_vector<int> compareid(totalnumoflines);

	typedef thrust::device_vector<int>::iterator IndexIterator;

	thrust::device_vector<int> indices(totalnumoflines);
	thrust::counting_iterator<int> indices_begin(0);
	thrust::counting_iterator<int> indices_end(totalnumoflines);

	// *********************************************************************************************************************

	thrust::transform(make_zip_iterator(make_tuple(dx_vec.begin(),dy_vec.begin(),dz_vec.begin())),
			  make_zip_iterator(make_tuple(dx_vec.end(),dy_vec.end(),dz_vec.end())),
			  compareid.begin(),
			  maxx_tuple());	
	//maxx_tuple() 若x偏导数的绝对值>y或z的偏导数的绝对值,返回1. (x,y,z)三元组
	IndexIterator dxindices_end = thrust::copy_if(indices_begin,indices_end,compareid.begin(),indices.begin(),thrust::identity<int>());
	std::cout<<"found "<< (dxindices_end - indices.begin()) << " x max values\n";
	int size1=(dxindices_end - indices.begin());
	
	thrust::copy(indices.begin(),dxindices_end,indexxmax);

	// *********************************************************************************************************************

	thrust::transform(make_zip_iterator(make_tuple(dx_vec.begin(),dy_vec.begin(),dz_vec.begin())),
			  make_zip_iterator(make_tuple(dx_vec.end(),dy_vec.end(),dz_vec.end())),
			  compareid.begin(),
			  maxy_tuple());	
	
	dxindices_end = thrust::copy_if(indices_begin,indices_end,compareid.begin(),indices.begin(),thrust::identity<int>());
	std::cout<<"found "<< (dxindices_end - indices.begin()) << " y max values\n";
	int size2=(dxindices_end - indices.begin());
	
	thrust::copy(indices.begin(),dxindices_end,indexymax);
	
	// *********************************************************************************************************************

	thrust::transform(make_zip_iterator(make_tuple(dx_vec.begin(),dy_vec.begin(),dz_vec.begin())),
			  make_zip_iterator(make_tuple(dx_vec.end(),dy_vec.end(),dz_vec.end())),
			  compareid.begin(),
			  maxz_tuple());	
	
	dxindices_end = thrust::copy_if(indices_begin,indices_end,compareid.begin(),indices.begin(),thrust::identity<int>());
	std::cout<<"found "<< (dxindices_end - indices.begin()) << " z max values\n";
	int size3=(dxindices_end - indices.begin());

	
	thrust::copy(indices.begin(),dxindices_end,indexzmax);

	
	sizen[0]=size1;	sizen[1]=size2;	sizen[2]=size3;

	// *********************************************************************************************************************	

	// they are saved into 3 vectors indexxmax[size1], indexymax[size2], indexzmax[size3]
		
	
	//thrust::copy(indices.begin(),dxindices_end,std::ostream_iterator<int>(std::cout,"\n"));
	
	
}
