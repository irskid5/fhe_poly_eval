# Parallel Polynomial Evaluation Algorithms for Leveled Fully Homomorphic Encryption

Evaluation of three parallel polynomial evaluation algorithms written for CUDA in C++ (Horner's method, Dorn's method, and Estrin's algorithm).

## Abstract

Computation in a Leveled Fully Homomorphic Encryption (LFHE) context is more inefficient than in a plaintext (PT) context. Latencies grow as operations on ciphertexts (CTs) grow, limiting the efficiency of larger computations like polynomial evaluations (PEs). In addition, the amount of successive CT multiplications is cryptographically bounded, limiting the maximum degree of polynomials that can be evaluated. As such, we explore three different polynomial evaluation algorithms for Leveled Fully Homomorphic Encryption (LFHE) with the following properties: algorithms are (1) parallelizable in a multicore environment to hide LFHE operation latency, and (2) strive to minimize CT-CT multiplicative depth to increase the amount of operations that can be applied afterwards and decrease the latency of operations. We show that Estrin’s algorithm is the best with high computational latencies, achieving a near linear speedup in execution time as the degree of polynomial grows. We show that Dorn’s algorithm becomes optimal as latencies approach plaintext levels.

## Repository

This repository includes the main kernel.cu file. This file evaluates the execuion time of Dorn's and Estrin's algorithms running on parallel CUDA threads with respect to Horner's method running on a single CUDA thread. Latency is introduced to plaintext operations since Microsoft SEAL, at the time of writing this README and report, has not been implemented on CUDA so we cannot test the evaluation algorithms on CUDA with SEAL. For more information and results, please review report.pdf.

The files in ./common are from the CUDA toolkit samples and used as helpers.

## Contributors

Vele Tosevski, B.A.Sc. in Engineering Science, M.A.Sc candidate

Faculty of Applied Science & Engineering, University of Toronto

vele.tosevski@mail.utoronto.ca