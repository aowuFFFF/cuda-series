#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

// modifiable
typedef float ft;
const int chunks = 64;
const size_t ds = 1024*1024*chunks;
const int count = 22;
const int num_streams = 8;

// not modifiable
const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;
__device__ float gpdf(float val, float sigma) {
  return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

__device__ double gpdf(double val, double sigma) {
  return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}

// compute average gaussian pdf value over a window around each point
/*
const ft * __restrict__ x：指向常量数据类型的指针 x，__restrict__ 告诉编译器这个指针不会在函数内部被alias（别名），有助于编译器优化代码。
ft * __restrict__ y：指向数据类型 ft 的指针 y，用于存储计算结果，同样使用 __restrict__。
const ft mean：高斯分布的均值。
const ft sigma：高斯分布的标准差。
const int n：输入数组 x 的长度。
*/
__global__ void gaussian_pdf(const ft * __restrict__ x, ft * __restrict__ y, const ft mean, const ft sigma, const int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    ft in = x[idx] - (count / 2) * 0.01f;
    ft out = 0;
    for (int i = 0; i < count; i++) {
      ft temp = (in - mean) / sigma;
      out += gpdf(temp, sigma);
      in += 0.01f;
    }
    y[idx] = out / count;
  }
}

// error check macro
#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

// host-based timing
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int main() {
  ft *h_x, *d_x, *h_y, *h_y1, *d_y;
  cudaHostAlloc(&h_x,  ds*sizeof(ft), cudaHostAllocDefault); //cudaHostAllocDefault：默认标志，分配的内存可以被主机和设备访问。
  cudaHostAlloc(&h_y,  ds*sizeof(ft), cudaHostAllocDefault);
  cudaHostAlloc(&h_y1, ds*sizeof(ft), cudaHostAllocDefault);
  cudaMalloc(&d_x, ds*sizeof(ft));
  cudaMalloc(&d_y, ds*sizeof(ft));
  cudaCheckErrors("allocation error");

  cudaStream_t streams[num_streams];    //cudaStream_t 是CUDA中用于表示流的数据类型
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);  //调用 cudaStreamCreate 函数创建一个新的流，并将其地址存储在 streams 数组的相应位置。
  }
  cudaCheckErrors("stream creation error");

  gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds); // warm-up

  for (size_t i = 0; i < ds; i++) {
    h_x[i] = rand() / (ft)RAND_MAX;
  }
  cudaDeviceSynchronize();  //当 CPU 调用 cudaDeviceSynchronize() 时，它会阻塞，直到 GPU 上所有先前启动的内核执行完毕。

  unsigned long long et1 = dtime_usec(0);

  cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
  gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds);
  cudaMemcpy(h_y1, d_y, ds * sizeof(ft), cudaMemcpyDeviceToHost);
  cudaCheckErrors("non-streams execution error");

  et1 = dtime_usec(et1);
  std::cout << "non-stream elapsed time: " << et1/(float)USECPSEC << std::endl;

#ifdef USE_STREAMS
  cudaMemset(d_y, 0, ds * sizeof(ft));

  unsigned long long et = dtime_usec(0); //调用 dtime_usec 函数开始计时，并将起始时间存储在变量 et 中

  for (int i = 0; i < chunks; i++) { //depth-first launch chunks 是要处理的数据块数量，循环将对每个数据块执行一系列操作。
    //使用 cudaMemcpyAsync 异步地从主机内存复制数据到设备内存。Async 表示操作是异步的，不会阻塞后续操作。
    //streams[i % num_streams] 指定了使用哪个流来执行这个传输操作。
    cudaMemcpyAsync(d_x + i * (ds / chunks), h_x + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyHostToDevice, streams[i % num_streams]);
    //核函数在指定的流上异步执行。
    gaussian_pdf<<<((ds / chunks) + 255) / 256, 256, 0, streams[i % num_streams]>>>(d_x + i * (ds / chunks), d_y + i * (ds / chunks), 0.0, 1.0, ds / chunks);
    //将计算结果从设备内存异步复制回主机内存。
    cudaMemcpyAsync(h_y + i * (ds / chunks), d_y + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyDeviceToHost, streams[i % num_streams]);
  }
  cudaDeviceSynchronize();
  cudaCheckErrors("streams execution error");

  et = dtime_usec(et); //结束计时并计算从开始到结束的总时间。

  for (int i = 0; i < ds; i++) { //检查 h_y 和 h_y1 数组是否完全相同，如果不同则输出错误信息并返回 -1。
    if (h_y[i] != h_y1[i]) {
      std::cout << "mismatch at " << i << " was: " << h_y[i] << " should be: " << h_y1[i] << std::endl;
      return -1;
    }
  }

  std::cout << "streams elapsed time: " << et/(float)USECPSEC << std::endl;
#endif

  return 0;
}
