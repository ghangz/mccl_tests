### Build
Before build mccl perf test, you need to install MACA SDK.
- Download MXMACA-SDK and decompress it.
- Find the file "mxmaca-sdk-install.sh" and run `bash mxmaca-sdk-install.sh`, the default install path is `/opt/maca`.

After install MACA SDK, run `bash build.sh` to build tests, the generated executable files are in the build directory.

### shell examples
  mccl.sh                Script for testing single-machine mccl                          Usage：bash mccl.sh gpu_num test_name
  cluster.sh             Script for testing multi-machine mccl                           Usage：bash cluster.sh ip_1 ip_2 ip_mask gpu_num test_name
  dragonfly.sh           Script for testing dragonfly network topology                   Usage：bash dragonfly.sh ip_1 ip_2 ip_mask gpu_num test_name
  mxccl_perf/mxccl.sh    Script for testing single-machine mxccl on METAX machines       Usage：bash mxccl.sh gpu_num
  nccl_perf/nccl.sh      Script for testing single-machine nccl on NVIDIA machines       Usage：bash nccl.sh gpu_num
  xccl.sh                Script for testing multi-machine heterogeneous mxccl and nccl   Usage：bash xccl.sh ip_1 ip_2 ip_mask gpu_num
  function/per_rank.sh   Script for testing different env config per rank or per node    Usage：called by other scripts
  function/mccl.sh       Single-machine mccl (use rankRun.sh instead of binary)          Usage：bash mccl.sh gpu_num
  function/cluster.sh    Multi-machine mccl (use rankRun.sh instead of binary)           Usage：bash cluster.sh ip_1 ip_2 ip_mask gpu_num

  Note that the default perf test files for mxccl_pef and nccl_perf are compiled based on nccl-tests v2.13.8.

***NOTE: you can refer to mccl.sh and cluster.sh for testing. The other scripts depend on MACA SDK, the path is `${MACA_PATH}/samples/mccl_tests/perf`.***

### Quick examples

Collect a small JSON environment report before performance runs:

```shell
export MACA_PATH=/opt/maca
./tools/collect_env.sh > mccl-env.json
```

Run with MPI on 4 processes (potentially on multiple nodes) with 1 GPUs each :
```shell
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib
${MACA_PATH}/ompi/bin/mpirun -n 4 --allow-run-as-root -mca pml ^ucx ./all_reduce_perf -b 8 -e 1G -f 2 -g 1 -d bfloat16
```

### Arguments

All tests support the same set of arguments :

* Number of GPUs
  * `-t,--nthreads <num threads>` number of threads per process. Default : 1.
  * `-g,--ngpus <GPUs per thread>` number of gpus per thread. Default : 1.
* Sizes to scan
  * `-b,--minbytes <min size in bytes>` minimum size to start with. Default : 32M.
  * `-e,--maxbytes <max size in bytes>` maximum size to end at. Default : 32M.
  * Increments can be either fixed or a multiplication factor. Only one of those should be used
    * `-i,--stepbytes <increment size>` fixed increment between sizes. Default : (max-min)/10.
    * `-f,--stepfactor <increment factor>` multiplication factor between sizes. Default : disabled.
* MCCL operations arguments
  * `-o,--op <sum/prod/min/max/avg/all>` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default : Sum.
  * `-d,--datatype <mccltype/all>` Specify which datatype to use. Default : Float.
  * `-r,--root <root/all>` Specify which root to use. Only for operations with a root like broadcast or reduce. Default : 0.
* Performance
  * `-n,--iters <iteration count>` number of iterations. Default : 20.
  * `-w,--warmup_iters <warmup iteration count>` number of warmup iterations (not timed). Default : 5.
  * `-m,--agg_iters <aggregation count>` number of operations to aggregate together in each iteration. Default : 1.
  * `-a,--average <0/1/2/3>` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.
* Test operation
  * `-p,--parallel_init <0/1>` use threads to initialize MCCL in parallel. Default : 0.
  * `-c,--check <0/1>` check correctness of results. This can be quite slow on large numbers of GPUs. Default : 1.
  * `-z,--blocking <0/1>` Make MCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
  * `-G,--cudagraph <num graph launches>` Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.
  * `-mca pml ^ucx` Skip ucx and speed up mpi.
  * `--allow-run-as-root` Allow the program to run as the root user, which is a common parameter for mpirun.
