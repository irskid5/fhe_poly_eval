// includes, system
#include <stdio.h>
#include <math.h>
#include <cuda/std/chrono>
using namespace cuda::std::chrono;

// includes CUDA Runtime
#include <cuda_runtime.h>

// includes, project
#include "common/inc/helper_cuda.h"
#include "common/inc/helper_functions.h" // helper utility functions 

using clock_value_t = long long;

void horner_on_cpu(float x, float* coeff, int n, float* ret);
__global__ void horner_kernel(float x, float* coeff, int n, float* ret, bool test_seal);
cudaError_t horner_on_cuda(float x, float* coeff, int n, float* ret, bool test_seal);
__global__ void nth_order_horner_child(float x_k, float* coeff, int k, int lastIdx, float* b, bool test_seal);
__global__ void nth_order_horner(float x, float* coeff, int k, int lastIdx, float* ret, bool test_seal);
cudaError_t nth_order_horner_on_cuda(float x, float* coeff, int k, int lastIdx, float* ret, bool test_seal);
__global__ void estrin_child(float x, float* coeff, float* coeff_cpy, bool odd, int threads, bool test_seal);
__global__ void estrin(float x, float* coeff, float* coeff_cpy, int threads, bool odd, float* ret, bool test_seal);
cudaError_t estrin_on_cuda(float x, float* coeff, int lastIdx, float* ret, bool test_seal);
__global__ void timer_kernel();
cudaError_t average_timings();

__device__ void mult_wait() {
    clock_value_t start_clk = clock64();
    clock_value_t cycles_elapsed;
    // full spec = 381664932
    // half spec = 190832466
    // quarter spec = 95416233
    // eigth spec = 47708116
    // 64th spec = 5963514
    // 1/100,000 spec = 3817
    do { cycles_elapsed = clock64() - start_clk; } while (cycles_elapsed < 3817);
    return;
}

__device__ void add_wait() {
    clock_value_t start_clk = clock64();
    clock_value_t cycles_elapsed;
    // full spec = 2472424
    // half spec = 1236212
    // quarter spec = 618106
    // eigth spec = 309053
    // 64th spec = 38631
    // 1/100,000 spec = 25
    do { cycles_elapsed = clock64() - start_clk; } while (cycles_elapsed < 25);
    return;
}

__global__ void horner_kernel(float x, float* coeff, int n, float* ret, bool test_seal)
{
    int i = n-1;
    float tmp = coeff[n];
    while (i >= 0) {
        tmp = tmp * x + coeff[i];

        // Seal test
        if (test_seal) {
            // mults (1 mult only at each timestep)
            mult_wait();

            // adds (1 add only at each timestep)
            add_wait();
        }

        i -= 1;
    }
    *ret = tmp;
}

__global__ void nth_order_horner_child(float x_k, float* coeff, int k, int lastIdx, float* b, bool test_seal) {
    // Calculate number of terms in shortened array
    int terms = 0;
    while (true) {
        if (terms * k + threadIdx.x > lastIdx) {
            break;
        }
        terms = terms + 1;
    }

    // Calculate subproblem results using Horner
    int i = terms - 2;
    float tmp = coeff[(terms-1)*k + threadIdx.x];
    while (i >= 0) {
        tmp = tmp * x_k + coeff[i*k + threadIdx.x];
        if (test_seal) {
            add_wait();
            mult_wait();
        }
        i -= 1;
    }
    b[threadIdx.x] = tmp;
}

__global__ void nth_order_horner(float x, float* coeff, int k, int lastIdx, float* ret, bool test_seal) {
    // Calculate x to the k 
    float x_k = powf(x, k);

    // Run horner's on k threads for k subcalculations
    float* b = (float*)malloc(k * sizeof(float));

    nth_order_horner_child<<< 1, k >>>(x_k, coeff, k, lastIdx, b, test_seal);
    cudaDeviceSynchronize();

    // Bring them together using horners
    int i = k - 2;
    float tmp = b[k-1];
    while (i >= 0) {
        tmp = tmp * x + b[i];
        if (test_seal) {
            add_wait();
            mult_wait();
        }
        i -= 1;
    }

    *ret = tmp;
    
    free(b);
}

__global__ void estrin_child(float x, float* coeff, float* coeff_cpy, bool odd, int threads, bool test_seal) {
    int startIdx = threadIdx.x * 2;
    if (odd) {
        if (threadIdx.x == threads - 1) {
            coeff_cpy[threadIdx.x] = coeff_cpy[startIdx];
            //printf("Thread: %d, val: %.2f\n", threadIdx.x, coeff_cpy[threadIdx.x]);
            return;
        }
    }
    coeff_cpy[threadIdx.x] = coeff_cpy[startIdx] + x * coeff_cpy[startIdx + 1];
    if (test_seal) {
        add_wait();
        mult_wait();
    }
    //printf("Thread: %d, val: %.2f\n", threadIdx.x, coeff_cpy[threadIdx.x]);
    return;
}

__global__ void estrin(float x, float* coeff, float* coeff_cpy, int threads, bool odd, float* ret, bool test_seal)
{
    float x_cpy = x;
    float tmp;
    while(threads > 1) {
        //printf("x: %.2f\n", x_cpy);
        estrin_child<<<1, threads>>>(x_cpy, coeff, coeff_cpy, odd, threads, test_seal);
        tmp = x_cpy * x_cpy;
        if (test_seal) {
            mult_wait();
        }
        cudaDeviceSynchronize();
        odd = (threads % 2 == 1) ? true : false;
        threads = (int)ceilf((float)threads / 2);
        x_cpy = tmp;
    }
    //printf("x: %.2f\n", x_cpy);
    *ret = coeff_cpy[0] + x_cpy * coeff_cpy[1];
    if (test_seal) {
        add_wait();
        mult_wait();
    }
    //printf("Thread: %d, val: %.2f\n", threadIdx.x, *ret);
}

__global__ void timer_kernel() {

    // test average time for an add and mult
    int i;
    long count = 1000;
    int add_durations = 0;
    int mult_durations = 0;
    int test_int = 10;
    for (i = 0; i < count; i++) {
        // reset
        test_int = 10;

        // test add
        auto start_add = high_resolution_clock::now();
        test_int = test_int * test_int;
        auto stop_add = high_resolution_clock::now();

        // test mult
        auto start_mult = high_resolution_clock::now();
        test_int = test_int + test_int;
        auto stop_mult = high_resolution_clock::now();

        // tally durations
        auto duration_add = duration_cast<microseconds>(stop_add - start_add);
        auto duration_mult = duration_cast<microseconds>(stop_mult - start_mult);
        add_durations += duration_add.count();
        mult_durations += duration_mult.count();
    }

    // print results
    printf("Average add = %.4f microseconds.\n", (float)add_durations/(float)count);
    printf("Average mult = %.4f microseconds.\n", (float)mult_durations / (float)count);

    // find optimal do nothing code for extending time of mults and adds
    int j;
0.02    long idle_count = 381664932; // for FHE mult (avg 37793 mult + 328987 relin = 366780 microsec)

    auto start = high_resolution_clock::now();

    clock_value_t start_clk = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start_clk; } while (cycles_elapsed < idle_count);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    printf("Idle time = %.2f microseconds for idle count %d.\n", (float)duration.count(), idle_count);
    //printf("%.2f\n", testInt);
}

void horner_on_cpu(float x, float* coeff, int n, float* ret) {
    int i = n - 1;
    float tmp = coeff[n];
    while (i >= 0) {
        tmp = tmp * x + coeff[i];
        i -= 1;
    }
    *ret = tmp;
}

cudaError_t horner_on_cuda(float x, float* coeff, int n, float* ret, bool test_seal)
{
    float* dev_coeff;
    float* dev_ret;
    std::ofstream horner_data;
    cudaError_t cudaStatus;

    int coeff_len = n + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers on GPU .
    cudaStatus = cudaMalloc((void**)&dev_coeff, coeff_len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ret, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_coeff, coeff, coeff_len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    horner_kernel <<<1, 1, 0, 0 >>> (x, dev_coeff, n, dev_ret, test_seal);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.8f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.8f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    horner_data.open("data.csv", std::ios_base::app);
    horner_data << gpu_time << ",";
    horner_data.close();

Error:
    cudaFree(dev_coeff);
    cudaFree(dev_ret);
    cudaFree(start);
    cudaFree(stop);

    return cudaStatus;
}

cudaError_t nth_order_horner_on_cuda(float x, float* coeff, int k, int lastIdx, float* ret, bool test_seal)
{
    float* dev_coeff;
    float* dev_ret;
    std::ofstream nth_horner_data;
    cudaError_t cudaStatus;

    int coeff_len = lastIdx + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers on GPU .
    cudaStatus = cudaMalloc((void**)&dev_coeff, coeff_len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ret, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_coeff, coeff, coeff_len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    nth_order_horner << <1, 1, 0, 0 >> > (x, dev_coeff, k, lastIdx, dev_ret, test_seal);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.8f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.8f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    nth_horner_data.open("data.csv", std::ios_base::app);
    nth_horner_data << gpu_time << ",";
    nth_horner_data.close();

Error:
    cudaFree(dev_coeff);
    cudaFree(dev_ret);
    cudaFree(start);
    cudaFree(stop);

    return cudaStatus;
}

cudaError_t estrin_on_cuda(float x, float* coeff, int lastIdx, float* ret, bool test_seal)
{
    float* dev_coeff;
    float* dev_coeff_cpy;
    float* dev_ret;
    std::ofstream estrin_data;
    cudaError_t cudaStatus;

    int coeff_len = lastIdx + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers on GPU .
    cudaStatus = cudaMalloc((void**)&dev_coeff, coeff_len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_coeff_cpy, coeff_len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ret, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_coeff, coeff, coeff_len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_coeff_cpy, coeff, coeff_len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    float gpu_time = 0.0f;

    int threads = (int)ceilf((float)coeff_len/2);
    bool odd = (coeff_len % 2 == 1) ? true : false;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    estrin<<<1, 1, 0, 0>>>(x, dev_coeff, dev_coeff_cpy, threads, odd, dev_ret, test_seal);
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // Restore copy of coeff
    cudaStatus = cudaMemcpy(dev_coeff_cpy, coeff, coeff_len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.8f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.8f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    estrin_data.open("data.csv", std::ios_base::app);
    estrin_data << gpu_time << "\n";
    estrin_data.close();

Error:

    cudaFree(dev_coeff);
    cudaFree(dev_coeff_cpy);
    cudaFree(dev_ret);
    cudaFree(start);
    cudaFree(stop);

    return cudaStatus;
}

cudaError_t average_timings()
{
    cudaError_t cudaStatus;

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed!");
        goto Error;
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    timer_kernel<<<1, 1>>>();
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }

    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.8f\n", gpu_time);
    printf("time spent by CPU in CUDA calls: %.8f\n", sdkGetTimerValue(&timer));
    printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

Error:
    cudaFree(start);
    cudaFree(stop);

    return cudaStatus;
}

int main(int argc, char* argv[])
{
    int devID;
    cudaDeviceProp deviceProps;
    cudaError_t cudaStatus;
    std::ofstream data;
    bool test_seal = true;

    // ready output file
    data.open("data.csv");
    data << "Horner,nth Horner,Estrin\n" << "\n";
    data.close();

    printf("[%s] - Starting...\n\n", argv[0]);
    printf("---------------------------------------- Basic Timings ----------------------------------------\n");

    cudaStatus = average_timings();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "timings failed!");
        return 1;
    }

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char**)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);

    printf("---------------------------------------- Main tests ----------------------------------------\n");
    srand(0);
    int degree;
    int maxDegree = 101;
    for (degree = 5; degree < maxDegree; degree++) {
        
        if (degree % 100 == 0) {
            printf("%.2f percent completed.\n", 100*(float)degree/(float)maxDegree);
        }
        
        // Init test values 
        //const int degree = 5;
        //float x = 2.0;
        //float coeff[degree + 1] = {1, 1, 1, 1, 1, 1 };


        float x = 0.5;
        float* coeff = (float*)malloc((degree + 1) * sizeof(float));
        int i;
        for (i = 0; i < degree + 1; i++) {
            coeff[i] = rand() / 100;
        }

        // k for nth horners
        //const int k = 1;
        const int k = (int)floor(degree / 2);

        float horner_output = 0;
        float nth_horner_output = 0;
        float estrin_output = 0;
        float cpu_output = 0;

        printf("---------------------------------------- Horner on CPU ----------------------------------------\n");
        horner_on_cpu(x, coeff, degree, &cpu_output);
        printf("Horner's Method Eval (CPU) = {%.4f}\n", cpu_output);

        printf("\n");

        printf("---------------------------------------- Horner on GPU ----------------------------------------\n");
        cudaStatus = horner_on_cuda(x, coeff, degree, &horner_output, test_seal);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "horner_on_cuda failed!");
            return 1;
        }
        printf("Horner's Method Eval (GPU) = {%.4f}\n", horner_output);
        if (horner_output != cpu_output)
            printf("!!!!!!!!!!!!! Horner WRONG !!!!!!!!!!!!\n");

        printf("\n");

        printf("-------------------------------------- Nth Horner on GPU --------------------------------------\n");
        cudaStatus = nth_order_horner_on_cuda(x, coeff, k, degree, &nth_horner_output, test_seal);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "nth_order_horner_on_cuda failed!");
            return 1;
        }
        printf("Nth-Order Horner's Method Eval (GPU) = {%.4f}\n", nth_horner_output);
        if (nth_horner_output != cpu_output)
            printf("!!!!!!!!!!!!! nth Horner WRONG !!!!!!!!!!!!\n");

        printf("\n");

        printf("-------------------------------------- Estrin on GPU --------------------------------------\n");
        cudaStatus = estrin_on_cuda(x, coeff, degree, &estrin_output, test_seal);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "estrin_on_cuda failed!");
            return 1;
        }
        printf("Estrin's Method Eval (GPU) = {%.4f}\n", estrin_output);
        if (estrin_output != cpu_output)
            printf("!!!!!!!!!!!!! Estrin WRONG !!!!!!!!!!!!\n");

        printf("\n");

        free(coeff);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    printf("Done!\n");

    system("pause");

    return 0;
}
