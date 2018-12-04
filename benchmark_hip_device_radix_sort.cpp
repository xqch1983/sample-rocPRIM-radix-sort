#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>
#include<stdlib.h>
#include <time.h> 
// Google Benchmark
// CmdParser
#include "cmdparser.hpp"

//#include "test_utils.hpp"
// HIP API
#include <hip/hip_runtime.h>
#include <hip/hip_hcc.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "device_radix_sort_hip.hpp"
#include "benchmark_utils.hpp"
#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }
#define output_to_verify 1 
#ifndef DEFAULT_N
const size_t DEFAULT_N = 1000 * 1000 *1 ;
//const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

#include <sys/time.h>

//#define HIP_CHECK(x) x

class Timer
{	
public:
void Start() { gettimeofday(&m_start, NULL); }
	void End() { gettimeofday(&m_end, NULL); }
	double  GetDelta()
	{
		return (m_end.tv_sec - m_start.tv_sec) * 1000.0 
			+ (m_end.tv_usec - m_start.tv_usec) / 1000.0;
	}
private:
	struct timeval m_start;
	struct timeval m_end;
};


namespace rp = rocprim;

const unsigned int batch_size = 10;
const unsigned int warmup_size = 5;

template<class Key, class Value>
void run_sort_pairs_benchmark3(hipStream_t stream, size_t size)
{
    using key_type = Key;
    using value_type = Value;
        Timer t;

    // Generate data
    std::vector<key_type> keys_input;
    if(std::is_floating_point<key_type>::value)
    {
        keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, size);
    }
    else
    {
        keys_input = get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max(),
            size
        );
    }
    std::vector<value_type> values_input(size);
    
   
    std::vector<value_type> values_output(size);

    std::vector<key_type> keys_output(size);

    std::iota(values_input.begin(), values_input.end(), 0);

#if 1 //init top 10 
    srand( (unsigned)time( NULL ) );
    for(int i =0;i<10;i++)
    {
	    keys_input[i] = random()%10*1.1;
	    values_input[i] = (i+1)*1.5;
    }
#endif
 
    for(int i=0;i<10;i++)
    	std::cout<<" num\t"<<i<<"\tkey "<<keys_input[i] <<"\tvalue\t"<<values_input[i]<<std::endl;

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    value_type * d_values_output;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    HIP_CHECK(
        rp::radix_sort_pairs(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, d_values_input, d_values_output, size,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );

    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::radix_sort_pairs(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    //for (auto _ : state)
   // {
       // auto start = std::chrono::high_resolution_clock::now();
	
	    t.Start();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::radix_sort_pairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
        HIP_CHECK(hipDeviceSynchronize());
	t.End();
        
	 double delta = t.GetDelta();
	std::cout<<"elsatp time \t"<< delta/10<<"ms"<<std::endl;	

// Getting results to host
    HIP_CHECK(
        hipMemcpy(
            keys_output.data(), d_keys_output,
            keys_output.size() * sizeof(typename decltype(keys_output)::value_type),
            hipMemcpyDeviceToHost
        )
    );

    HIP_CHECK(
        hipMemcpy(
            values_output.data(), d_values_output,
            values_output.size() * sizeof(typename decltype(values_output)::value_type),
            hipMemcpyDeviceToHost
        )
    );
#if 1 
    std::cout<<"\n\n output result:\n";
    for(int i=0;i<10;i++)
    {
	    std::cout<<" output:  num "<<i<<"\tkey "<<keys_output[i] <<"\tvalue\t"<<values_output[i]<<std::endl;
    }
#endif
    //auto end = std::chrono::high_resolution_clock::now();
        //auto elapsed_seconds =
            //std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
	    //printf("th eelsapse thime %10.f\n",elapsed_seconds);
	   //std::cout<<elapsed_seconds; 
      //  state.SetIterationTime(elapsed_seconds.count());
    //}
   // state.SetBytesProcessed(
    //    state.iterations() * batch_size * size * (sizeof(key_type) + sizeof(value_type))
    //);
    //state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_values_output));
}

template<class Key, class Value>
void run_sort_pairs_benchmark2(hipStream_t stream, size_t size)
{
    using key_type = Key;
    using value_type = Value;
        Timer t;

    // Generate data
    std::vector<key_type> keys_input(size);
    std::vector<value_type> values_input(size);
    for(int i=0;i<keys_input.size();i++){
	keys_input[i]=i+1;
        values_input[i]=keys_input.size()-i*1.1; 
   }
#if 0 
   if(std::is_floating_point<key_type>::value)
    {
        keys_input = get_random_data<key_type>(size, (key_type)-1000, (key_type)+1000, size);
    }
    else
    {
        keys_input = get_random_data<key_type>(
            size,
            std::numeric_limits<key_type>::min(),
            std::numeric_limits<key_type>::max(),
            size
        );
    }
#endif
    //std::iota(values_input.begin(), values_input.end(), 0);

    for(int i=0;i<10;i++)
    std::cout<<" num\t"<<i<<"\tkey "<<keys_input[i] <<"\tvalue"<<values_input[i]<<std::endl;

    key_type * d_keys_input;
    key_type * d_keys_output;
    HIP_CHECK(hipMalloc(&d_keys_input, size * sizeof(key_type)));
    HIP_CHECK(hipMalloc(&d_keys_output, size * sizeof(key_type)));
    HIP_CHECK(
        hipMemcpy(
            d_keys_input, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyHostToDevice
        )
    );

    value_type * d_values_input;
    value_type * d_values_output;
    HIP_CHECK(hipMalloc(&d_values_input, size * sizeof(value_type)));
    HIP_CHECK(hipMalloc(&d_values_output, size * sizeof(value_type)));
    HIP_CHECK(
        hipMemcpy(
            d_values_input, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyHostToDevice
        )
    );

    void * d_temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
#if 1
    HIP_CHECK(
        rp::radix_sort_pairs(
            d_temporary_storage, temporary_storage_bytes,
            d_keys_input, d_keys_output, d_values_input, d_values_output, size,
            0, sizeof(key_type) * 8,
            stream, false
        )
    );
#endif
    HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());
#if 1
    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        HIP_CHECK(
            rp::radix_sort_pairs(
                d_temporary_storage, temporary_storage_bytes,
                d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                0, sizeof(key_type) * 8,
                stream, false
            )
        );
    }
#endif
    HIP_CHECK(hipDeviceSynchronize());

    //for (auto _ : state)
   // {
       // auto start = std::chrono::high_resolution_clock::now();
#if 1	
	    t.Start();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(
                rp::radix_sort_pairs(
                    d_temporary_storage, temporary_storage_bytes,
                    d_keys_input, d_keys_output, d_values_input, d_values_output, size,
                    0, sizeof(key_type) * 8,
                    stream, false
                )
            );
        }
#endif     
   HIP_CHECK(hipDeviceSynchronize());
	t.End();
        
	 double delta = t.GetDelta();
	std::cout<<"elsatp time \t"<< delta<<std::endl;	

     hipMemcpy(
            d_keys_output, keys_input.data(),
            size * sizeof(key_type),
            hipMemcpyDeviceToHost
        );

     hipMemcpy(
            d_values_output, values_input.data(),
            size * sizeof(value_type),
            hipMemcpyDeviceToHost
        );

	std::cout<<"2elsatp time \t"<< delta<<std::endl;	

    HIP_CHECK(hipFree(d_temporary_storage));
    HIP_CHECK(hipFree(d_keys_input));
    HIP_CHECK(hipFree(d_keys_output));
    HIP_CHECK(hipFree(d_values_input));
    HIP_CHECK(hipFree(d_values_output));
}



int main(int argc, char *argv[])
{
	cli::Parser parser(argc, argv);
	parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
	parser.set_optional<int>("trials", "trials", -1, "number of iterations");
	parser.run_and_exit_if_error();

	// Parse argv
	//benchmark::Initialize(&argc, argv);
	const size_t size = parser.get<size_t>("size");
	const int trials = parser.get<int>("trials");

	//const size_t size = 1000000;
	//const int trials = 1;

	std::cout<<size<<std::endl;; 
	std::cout<<trials<<std::endl; 
	// HIP
	//
	hipStream_t stream = 0; // default
	hipDeviceProp_t devProp;
	int device_id = 0;
	HIP_CHECK(hipGetDevice(&device_id));
	HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
	std::cout << "[HIP] Device name: " << devProp.name << std::endl;
	//run_sort_pairs_benchmark2( );
	run_sort_pairs_benchmark3<float, int>( stream, size);  
	//run_sort_pairs_benchmark3<float, float>( stream, size);  
	run_sort_pairs_benchmark3<int, float>( stream, size);  
	//run_sort_pairs_benchmark3<int, int>( stream, size);  

	std::cout<<"........... successfully!\n";

	return 0;



}

