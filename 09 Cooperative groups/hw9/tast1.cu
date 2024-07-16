#include <cooperative_groups.h>
#include <stdio.h>
using namespace cooperative_groups;  //协同组的命名空间

const int nTPB = 256;  //定义每个线程块的线程数为256。

//这个函数实现了一个简单的归约操作，将线程组内的数据归约为一个单一的值。
//thread_group:是CUDA中的一个类，代表一组线程，可以用来进行线程之间的同步和数据共享。
//x：传入一个指向共享内存的指针
//val：针对每个线程组内的线程进行规约
__device__ int reduce(thread_group g, int *x, int val) { //
  int lane = g.thread_rank();        //g.thread_rank()调用返回当前线程在其所属线程组中的索引。
  for (int i = g.size()/2; i > 0; i /= 2) {
    x[lane] = val;       g.sync();    //sync方法确保线程组内所有线程到达这一点后才继续执行。
    if (lane < i) val += x[lane + i];  g.sync();
  }
  if (g.thread_rank() == 0) printf("group partial sum: %d\n", val);
  return val;
}

//定义了一个内核函数，它将在GPU上并行执行。
__global__ void my_reduce_kernel(int *data){

  __shared__ int sdata[nTPB];  //声明了一块大小为nTPB的共享内存，供线程块内的所有线程使用。
  // task 1a: create a proper thread block group below：
  auto g1 = FIXME
  size_t gindex = g1.group_index().x * nTPB + g1.thread_index().x;
  // task 1b: uncomment and create a proper 32-thread tile below, using group g1 created above
  // auto g2 = FIXME 
  // task 1c: uncomment and create a proper 16-thread tile below, using group g2 created above
  // auto g3 = FIXME
  // for each task, adjust the group to point to the last group created above
  auto g = FIXME
  
  // Make sure we send in the appropriate patch of shared memory
  //根据当前线程的索引和最终选择的线程组的大小，计算共享内存中的偏移量。
  int sdata_offset = (g1.thread_index().x / g.size()) * g.size();  
  reduce(g, sdata + sdata_offset, data[gindex]);  //使用计算得到的偏移量，调用归约函数，将归约操作的结果存储在共享内存的适当位置。
}

//在主函数中，分配内存，初始化数据，启动内核函数，并同步设备以等待内核执行完成。
int main(){

  int *data;
  cudaMallocManaged(&data, nTPB*sizeof(data[0]));
  for (int i = 0; i < nTPB; i++) data[i] = 1;
  
  my_reduce_kernel<<<1,nTPB>>>(data);
  
  cudaError_t err = cudaDeviceSynchronize();  ////当 CPU 调用 cudaDeviceSynchronize() 时，它会阻塞，直到 GPU 上所有先前启动的内核执行完毕。
  if (err != cudaSuccess) printf("cuda error: %s\n", cudaGetErrorString(err));
}
