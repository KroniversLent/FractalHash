# FractalSponge-256 build system
# Fix for linker error: fractal_sponge.c is compiled with nvcc -x c++
# so all objects share the same C++ ABI. extern "C" in fractal_sponge.h
# prevents name mangling on the public API symbols.

CC      := gcc
# -march=native omitted: enables -mfma on modern CPUs, breaking cross-platform
# reproducibility. fractal_sponge.c uses #pragma STDC FP_CONTRACT OFF instead.
CFLAGS  := -O2 -Wall -Wextra -lm -ffp-contract=off
NVCC    := nvcc

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
              | head -1 | tr -d '.' | awk '{printf "sm_%s", $$1}')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := sm_75
endif

NVCCFLAGS := -O3 -arch=$(GPU_ARCH) --use_fast_math -Xcompiler -O2 \
             -lineinfo -Wno-deprecated-gpu-targets

CPU_OBJS    := fractal_sponge.o fractal_sponge_avx2.o main.o
CIPHER_OBJS := fractal_sponge.o fractal_sponge_avx2.o fractal_cipher.o cipher_main.o

.PHONY: all cpu gpu cipher clean test test-gpu test-cipher info

all: cpu cipher gpu
cpu: fractal_hash
cipher: fractal_cipher
gpu: fractal_gpu

# ── CPU ──────────────────────────────────────────────────────────────────────
fractal_hash: $(CPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

fractal_sponge.o: fractal_sponge.c fractal_sponge.h
	$(CC) $(CFLAGS) -c -o $@ $<

# AVX2 file compiled with -mavx2 only — still strict IEEE 754, no fast-math
fractal_sponge_avx2.o: fractal_sponge_avx2.c fractal_sponge.h
	$(CC) $(CFLAGS) -mavx2 -c -o $@ $<

main.o: main.c fractal_sponge.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Cipher ───────────────────────────────────────────────────────────────────
fractal_cipher: $(CIPHER_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

fractal_cipher.o: fractal_cipher.c fractal_cipher.h fractal_sponge.h
	$(CC) $(CFLAGS) -c -o $@ $<

cipher_main.o: cipher_main.c fractal_cipher.h fractal_sponge.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ── GPU ──────────────────────────────────────────────────────────────────────
# fractal_sponge_cpu.o: compiled WITHOUT --use_fast_math.
# --use_fast_math alters IEEE 754 rounding/FMA on host code, making
# fractal_gpu produce different digests than fractal_hash for the same input.
# NVCCFLAGS_STRICT omits --use_fast_math so both binaries are identical.
NVCCFLAGS_STRICT := -O3 -arch=$(GPU_ARCH) -Xcompiler -O2 \
                    -lineinfo -Wno-deprecated-gpu-targets \
                    -diag-suppress 177

fractal_gpu: fractal_sponge_cu.o birthday.o avalanche.o differential.o gpu_main.o fractal_sponge_cpu.o fractal_sponge_avx2_cpu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lm

fractal_sponge_cpu.o: fractal_sponge.c fractal_sponge.h
	$(NVCC) $(NVCCFLAGS_STRICT) -x c++ -o $@ -c $<

# AVX2 host code compiled via nvcc so it shares the same C++ ABI as
# fractal_sponge_cpu.o.  -Xcompiler -mavx2 passes the flag to the host gcc.
fractal_sponge_avx2_cpu.o: fractal_sponge_avx2.c fractal_sponge.h
	$(NVCC) $(NVCCFLAGS_STRICT) -Xcompiler -mavx2 -x c++ -o $@ -c $<

fractal_sponge_cu.o: fractal_sponge.cu fractal_sponge.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

birthday.o: birthday.cu fractal_sponge.cuh fractal_sponge.h
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

avalanche.o: avalanche.cu fractal_sponge.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

differential.o: differential.cu fractal_sponge.cuh
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

gpu_main.o: gpu_main.cu fractal_sponge.cuh fractal_sponge.h
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

# ── Tests ────────────────────────────────────────────────────────────────────
test: fractal_hash fractal_cipher
	@echo "=== CPU test vectors ===" && ./fractal_hash --test
	@echo "" && echo "=== CPU bench ===" && ./fractal_hash --bench
	@echo "" && echo "=== Cipher self-test ===" && ./fractal_cipher test

test-cipher: fractal_cipher
	@./fractal_cipher test

test-gpu: fractal_gpu
	@echo "=== GPU bench ===" && ./fractal_gpu --bench
	@echo "" && echo "=== GPU avalanche N=4096 ===" && ./fractal_gpu --avalanche 4096
	@echo "" && echo "=== GPU birthday ===" && ./fractal_gpu --birthday

# ── Utility ──────────────────────────────────────────────────────────────────
clean:
	rm -f fractal_hash fractal_gpu fractal_cipher \
	      fractal_sponge.o fractal_sponge_avx2.o main.o \
	      fractal_cipher.o cipher_main.o \
	      fractal_sponge_cpu.o fractal_sponge_avx2_cpu.o fractal_sponge_cu.o \
	      birthday.o avalanche.o differential.o gpu_main.o

info:
	@echo "GPU arch detected: $(GPU_ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total \
	  --format=csv,noheader 2>/dev/null || echo "(no GPU found)"
