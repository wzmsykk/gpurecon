#include"headerfiles.h"
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
struct saxbc_functor_gpu
{
	float a;
	float b;
	float c;
public:
	saxbc_functor_gpu(float a_, float b_, float c_)
	{
		a = a_;
		b = b_;
		c = c_;
	}

	__host__ __device__
	float operator()(const float& x) const
	{
		return (a * x + b) / c;
	}
};
struct roundsaxbc_functor
{
	float a;
	float b;
	float c;
public:
	roundsaxbc_functor(float a_, float b_, float c_)
	{
		a = a_;
		b = b_;
		c = c_;
	}

	__host__ __device__
	int operator()(const float& x) const
	{
		return floor((a * x + b) / c);
	}
};
struct readmatrix_YZX_functor
{
	//template <typename Tuple>
	int xdim, ydim, zdim;
	float* attu_matrix;
	readmatrix_YZX_functor(float* attu_matrix_, int xdim_, int ydim_, int zdim_) {
		attu_matrix = attu_matrix_;
		xdim = xdim_;
		ydim = ydim_;
		zdim = zdim_;

	}

	__host__ __device__
	float operator()(thrust::tuple<int, int, int> t)
	{
		// t(0)=x,t(1)=y, t(2)=z, t(3)=out
		return  attu_matrix[thrust::get<1>(t) * zdim * xdim + thrust::get<2>(t) * xdim + thrust::get<0>(t)];
	}
};
struct readmatrix_ZYX_functor
{
	int xdim, ydim, zdim;
	float* attu_matrix;
	readmatrix_ZYX_functor(float* attu_matrix_, int xdim_, int ydim_, int zdim_) {
		attu_matrix = attu_matrix_;
		xdim = xdim_;
		ydim = ydim_;
		zdim = zdim_;
	}

	__host__ __device__
	float operator()(thrust::tuple<int, int, int> t)
	{
		// t(0)=x, t(1)=y, t(2)=z, t(3)=out
		//border check 
		//在外部均取为0
		int xx, yy, zz;
		xx = thrust::get<0>(t);
		yy = thrust::get<1>(t);
		zz = thrust::get<2>(t);
		if (xx >= xdim || xx < 0) {
			return 0;
		}
		if (yy >= xdim || yy < 0) {
			return 0;
		}
		if (zz >= xdim || zz < 0) {
			return 0;
		}
		
		return attu_matrix[zz * ydim * xdim + yy * xdim + xx];
	}
};
int batchcorr(float* lines, int linesN, CTdims* ctdim, float* attenuation_matrix) {
	for (int i = 0; i < linesN; ++i) {
		CUDAlor* the_line = (CUDAlor*)lines + i;
#ifdef DEBUG
		if (i == 0) {
			attenucorrxyz((CUDAlor*)the_line, ctdim, attenuation_matrix, 2);
		}
		else {
#endif // DEBUG
			attenucorrxyz((CUDAlor*)the_line, ctdim, attenuation_matrix, 0);
#ifdef DEBUG
		}
#endif // DEBUG	
	}
	return 0;
}
int attenucorrxyz(CUDAlor* lines, CTdims* ctdim, float* attenuation_matrix,int dbglv) {


#ifdef DEBUG
	int debugLevel = 2;
	debugLevel = dbglv;
	//0 None 1 Partial 2 Full 
#endif // DEBUG

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
	float atf = 0;
	//CT surface 是从0-Nx 共Nx+1个
	//float x0, x1, y0, y1, z0, z1;


		// convert pointer type to struct
	CUDAlor* the_line = (CUDAlor*)lines;


	//计算衰减参数		
	float x0 = the_line->rx0;//lor线端点x0
	float x1 = the_line->rx1;//lor线端点x1
	float y0 = the_line->ry0;//lor线端点y0
	float y1 = the_line->ry1;//lor线端点y1
	float z0 = the_line->rz0;//lor线端点z0
	float z1 = the_line->rz1;//lor线端点z1
	float D = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0));
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("lor线端点P_0=(%lf,%lf,%lf)\n", x0, y0, z0);
		printf("lor线端点P_1=(%lf,%lf,%lf)\n", x1, y1, z1);
		printf("CT 尺寸Nx=%d,Ny=%d,Nz=%d\n", ctnx, ctny, ctnz);
		printf("CT 最小端点(0,0,0)在实际空间位置(in mm)(OFFSET)=(%lf,%lf,%lf)\n", ctx0, cty0, ctz0);
		printf("CT像素尺寸xyzspace=(%lf,%lf,%lf)\n", xspace, yspace, zspace);
	}
#endif // DEBUG
	float alphax0, alphay0, alphaz0;
	float alphaxn, alphayn, alphazn;
	float alphamin, alphamax;
	int imin, jmin, kmin;
	int imax, jmax, kmax;
	the_line->attcorrvalue = 1.0;
	bool calc_x = true, calc_y = true, calc_z = true;
	//判断LOR在CT内
#ifdef DEBUG
	if (debugLevel > 0) {
		if (x0 > ctx0 && x0<ctxn && y0>cty0 && y0<ctyn && z0>ctz0 && z0 < ctzn) {

			printf("(DEBUG) LOR两端点在CT内部\n");

		}
	}
#endif // DEBUG
	if (x0 - x1 == 0) {
		calc_x = false;
#ifdef DEBUG
		if (debugLevel > 0) {
			printf("X向不变\n");
		}
#endif // DEBUG
	}

	if (y0 - y1 == 0)
	{
		calc_y = false;
#ifdef DEBUG
		if (debugLevel > 0) {
			printf("Y向不变\n");
		}
#endif // DEBUG
	}
	if (z0 - z1 == 0)
	{
		calc_z = false;
#ifdef DEBUG
		if (debugLevel > 0) {
			printf("Z向不变\n");
		}
#endif // DEBUG
	}
	if (!calc_x && !calc_y && !calc_z) {
#ifdef DEBUG
		if (debugLevel > 0) {
			printf("是一个点？？\n");
		}
#endif // DEBUG
		atf = 0;
		the_line->attcorrvalue = 1.0;
		return 0;
	}
	thrust::device_vector<float>amaxvec;
	amaxvec.push_back(1.0f);
	thrust::device_vector<float>aminvec;
	aminvec.push_back(0.0f);

	if (calc_x) {
		alphax0 = (ctx0 - x0) / (x1 - x0);
		alphaxn = (ctxn - x0) / (x1 - x0);
		aminvec.push_back(thrust::min(alphax0, alphaxn));
		amaxvec.push_back(thrust::max(alphax0, alphaxn));
	}
	if (calc_y) {
		alphay0 = (cty0 - y0) / (y1 - y0);
		alphayn = (ctyn - y0) / (y1 - y0);
		aminvec.push_back(thrust::min(alphay0, alphayn));
		amaxvec.push_back(thrust::max(alphay0, alphayn));
	}
	if (calc_z) {
		alphaz0 = (ctz0 - z0) / (z1 - z0);
		alphazn = (ctzn - z0) / (z1 - z0);
		aminvec.push_back(thrust::min(alphaz0, alphazn));
		amaxvec.push_back(thrust::max(alphaz0, alphazn));
	}

	//thrust::maximum<float> tmax;
	//thrust::minimum<float> tmin;
	alphamin = *thrust::max_element(aminvec.begin(), aminvec.end());
	alphamax = *thrust::min_element(amaxvec.begin(), amaxvec.end());
#ifdef DEBUG
	if (debugLevel > 1) {
		if (calc_x) {
			printf("alphax0=%lf,alphaxn=%lf\n", alphax0, alphaxn);
		}

		printf("alphamin=%lf,alphamax=%lf\n", alphamin, alphamax);
	}
#endif//DEBUG


	if (alphamax <= alphamin) {
		//no interaction with CT
		atf = 0;
		the_line->attcorrvalue = 1.0;
#ifdef DEBUG
		if (debugLevel > 0) {
			printf("LOR与CT不相交\n");
		}
#endif//DEBUG
		return 0;
	}

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
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("imin=%d,imax=%d\n", imin, imax);
		printf("jmin=%d,jmax=%d\n", jmin, jmax);
		printf("kmin=%d,kmax=%d\n", kmin, kmax);
	}
#endif//DEBUG
	thrust::device_vector<float> ivec((imax - imin + 1), 0.0f);
	thrust::device_vector<float> jvec((jmax - jmin + 1), 0.0f);
	thrust::device_vector<float> kvec((kmax - kmin + 1), 0.0f);
	thrust::device_vector<float> ax((imax - imin + 1), 0.0f);
	thrust::device_vector<float> ay((jmax - jmin + 1), 0.0f);
	thrust::device_vector<float> az((kmax - kmin + 1), 0.0f);
	if (x1 - x0 >= 0) {
		thrust::sequence(thrust::device,ivec.begin(), ivec.end(), imin);
	}
	else {
		thrust::sequence(thrust::device, ivec.rbegin(), ivec.rend(), imin);
	}
	if (y1 - y0 >= 0) {
		thrust::sequence(thrust::device, jvec.begin(), jvec.end(), jmin);
	}
	else {
		thrust::sequence(thrust::device, jvec.rbegin(), jvec.rend(), jmin);
	}
	if (z1 - z0 >= 0) {
		thrust::sequence(thrust::device, kvec.begin(), kvec.end(), kmin);
	}
	else {
		thrust::sequence(thrust::device, kvec.rbegin(), kvec.rend(), kmin);
	}

	thrust::transform(ivec.begin(), ivec.end(), ax.begin(), saxbc_functor_gpu(xspace, (ctx0 - x0), (x1 - x0)));
	thrust::transform(jvec.begin(), jvec.end(), ay.begin(), saxbc_functor_gpu(yspace, (cty0 - y0), (y1 - y0)));
	thrust::transform(kvec.begin(), kvec.end(), az.begin(), saxbc_functor_gpu(zspace, (ctz0 - z0), (z1 - z0)));
	//SAXBC=(A*X+B)/C
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("合并前alpha_vec_x：\n");
		thrust::copy(ax.begin(), ax.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("合并前alpha_vec_y：\n");
		thrust::copy(ay.begin(), ay.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("\n");
		printf("合并前alpha_vec_z：\n");
		thrust::copy(az.begin(), az.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("\n");
	}
#endif//DEBUG
	thrust::device_vector<float>::iterator midit1bg, midit1nd, midit2bg, midit2nd;
	
	thrust::device_vector<float> mergeresult;
	if (calc_x && calc_y && calc_z) {
		//merge XY first
		thrust::device_vector<float> tempresult(ax.size()+ay.size());
		thrust::merge(ax.begin(), ax.end(), ay.begin(), ay.end(), tempresult.begin());
		midit1bg = tempresult.begin();
		midit1nd = tempresult.end();
		midit2bg = az.begin();
		midit2nd = az.end();
		mergeresult.resize(tempresult.size() + az.size());
		thrust::merge(midit1bg, midit1nd, midit2bg, midit2nd,mergeresult.begin());
	}
	else if (calc_x && calc_y && !calc_z)
	{
		midit1bg = ax.begin();
		midit1nd = ax.end();
		midit2bg = ay.begin();
		midit2nd = ay.end();
		mergeresult.resize(ax.size() + ay.size());
		thrust::merge(midit1bg, midit1nd, midit2bg, midit2nd, mergeresult.begin());
	}
	else if (!calc_x && calc_y && calc_z)
	{
		midit1bg = ay.begin();
		midit1nd = ay.end();
		midit2bg = az.begin();
		midit2nd = az.end();
		mergeresult.resize(ay.size() + az.size());
		thrust::merge(midit1bg, midit1nd, midit2bg, midit2nd, mergeresult.begin());
	}
	else if (calc_x && !calc_y && calc_z)
	{
		midit1bg = ax.begin();
		midit1nd = ax.end();
		midit2bg = az.begin();
		midit2nd = az.end();
		mergeresult.resize(ax.size() + az.size());
		thrust::merge(midit1bg, midit1nd, midit2bg, midit2nd, mergeresult.begin());
	}
	else if (!calc_x && !calc_y && calc_z)
	{
		mergeresult.resize(az.size());
		thrust::copy(az.begin(), az.end(),mergeresult.begin());
	}
	else if (!calc_x && calc_y && !calc_z)
	{
		mergeresult.resize(ay.size());
		thrust::copy(ay.begin(), ay.end(), mergeresult.begin());
	}
	else if (calc_x && !calc_y && !calc_z)
	{
		mergeresult.resize(ax.size());
		thrust::copy(ax.begin(), ax.end(), mergeresult.begin());
	}
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("插入首尾前alpha_vec：\n");
		thrust::copy(mergeresult.begin(), mergeresult.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("\n");
	}
#endif//DEBUG
	mergeresult.insert(mergeresult.begin(), alphamin);
	mergeresult.insert(mergeresult.end(), alphamax);
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("合并后alpha_vec：\n");
		thrust::copy(mergeresult.begin(), mergeresult.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("\n");
	}
#endif//DEBUG
	auto it = thrust::unique(mergeresult.begin(), mergeresult.end());
	mergeresult.resize(thrust::distance(mergeresult.begin(), it));//alpha去重
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("去重后alpha_vec：\n");
		thrust::copy(mergeresult.begin(), mergeresult.end(), std::ostream_iterator<float>(std::cout, ","));
		printf("\n");
	}
#endif//DEBUG
	//从此得到所有与CT网格边界交点
	size_t totalpts = mergeresult.size();


	thrust::device_vector<float> l(totalpts);
	thrust::device_vector<float> lfin(totalpts - 1);
	thrust::adjacent_difference(mergeresult.begin(), mergeresult.end(), l.begin());
	l.erase(l.begin());

	thrust::device_vector<float> Darray(totalpts - 1);
	thrust::fill(Darray.begin(), Darray.end(), D);
	thrust::transform(l.begin(), l.end(), Darray.begin(), lfin.begin(), thrust::multiplies<float>());
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("LOR在各个体素中路径长分别为：\n");
		for (auto iter = lfin.begin(); iter != lfin.end(); ++iter) {
			thrust::copy_n(iter, 1, std::ostream_iterator<float>(std::cout, ","));
		}
	}
#endif//DEBUG

	//从此得到LOR经过单个体素内线段长度
	thrust::device_vector<int> mx(totalpts - 1);
	thrust::device_vector<int> my(totalpts - 1);
	thrust::device_vector<int> mz(totalpts - 1);

	thrust::device_vector<float> adjsuma(totalpts);
	thrust::device_vector<float> adjmeana(totalpts);

	thrust::adjacent_difference(mergeresult.begin(), mergeresult.end(), adjsuma.begin(), thrust::plus<float>());
	thrust::device_vector <float > literal2(totalpts);
	thrust::fill(literal2.begin(), literal2.end(), 2.0f);

	thrust::transform(adjsuma.begin(), adjsuma.end(), literal2.begin(), adjmeana.begin(), thrust::divides<float>());//得到临近均值


	thrust::transform(adjmeana.begin() + 1, adjmeana.end(), mx.begin(), roundsaxbc_functor((x1 - x0), (x0 - ctx0), xspace));
	thrust::transform(adjmeana.begin() + 1, adjmeana.end(), my.begin(), roundsaxbc_functor((y1 - y0), (y0 - cty0), yspace));
	thrust::transform(adjmeana.begin() + 1, adjmeana.end(), mz.begin(), roundsaxbc_functor((z1 - z0), (z0 - ctz0), zspace));
	auto first = thrust::make_zip_iterator(thrust::make_tuple(mx.begin(), my.begin(), mz.begin()));
	auto last = thrust::make_zip_iterator(thrust::make_tuple(mx.end(), my.end(), mz.end()));


	//round((a*x+b)/c)
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("经过CT体素：\n");
		for (auto iter = first; iter != last; ++iter) {
			std::cout <<"("<< thrust::get<0>(*iter)<<","<< thrust::get<1>(*iter)<< ","<<thrust::get<2>(*iter)<<")\n";
			
		}
	}
#endif//DEBUG
	
	thrust::device_vector<float> voxelvalue(totalpts - 1);
	first = thrust::make_zip_iterator(thrust::make_tuple(mx.begin(), my.begin(), mz.begin()));
	last = thrust::make_zip_iterator(thrust::make_tuple(mx.end(), my.end(), mz.end()));

	thrust::transform(first, last, voxelvalue.begin(), readmatrix_ZYX_functor(attenuation_matrix, ctnx, ctny, ctnz));

#ifdef DEBUG
	if (debugLevel > 1) {
		printf("各个体素权重分别为：\n");
		for (auto iter = voxelvalue.begin(); iter != voxelvalue.end(); ++iter) {
			thrust::copy_n(iter, 1, std::ostream_iterator<float>(std::cout, ","));
		}
	}
#endif//DEBUG
	atf = thrust::inner_product(voxelvalue.begin(), voxelvalue.end(), lfin.begin(), 0.0f);
#ifdef DEBUG
	if (debugLevel > 1) {
		printf("内积结果为%lf\n", atf);
	}
#endif//DEBUG
	if (atf > 0) {
		float cv = exp(-atf);
		the_line->attcorrvalue = cv;
	}
	else {
		the_line->attcorrvalue = 1.0;
	}
	return 0;


}