# CUDA 编程实战 —— 示例 Makefile
# 用法: make            # 编译全部
#       make run        # 编译并运行全部（结果写入 results/）
#       make clean
# 目标架构：H20 = sm_90
ARCH ?= sm_90
NVCC ?= nvcc
FLAGS ?= -O3 -arch=$(ARCH) -lineinfo
BIN := bin

SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,$(BIN)/%,$(SRCS))

.PHONY: all run clean dirs

all: dirs $(BINS)

dirs:
	@mkdir -p $(BIN) results

$(BIN)/%: %.cu | dirs
	$(NVCC) $(FLAGS) $< -o $@

run: all
	@mkdir -p results
	@for b in $(BINS); do \
		name=$$(basename $$b); \
		echo "===== run $$name ====="; \
		./$$b | tee results/$$name.txt; \
	done

clean:
	rm -rf $(BIN) results *.o *.cubin *.ptx
