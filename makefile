CXX := g++
CXXFLAGS := -std=c++17 -Wall  -g
LDFLAGS := 
SRC_DIR := test_bed/src
BUILD_DIR := build
BIN := cppDL

USE_AVX256 ?= 1
USE_VULKAN ?= 0 
USE_OPENGL ?= 0 

ifeq ($(USE_AVX256),1)
	CXXFLAGS += -DUSE_AVX256 -Ofast -march=native -flto -mavx2 -mfma -fopenmp
	LDFLAGS += -DUSE_AVX256 -fopenmp
else 
	CXXFLAGS += -DUSE_AVX256=0
endif

SRCS := $(wildcard $(SRC_DIR)/*.cpp)

OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean

all: $(BIN)
	@echo "**EXECUTING**"
	@./$(BIN)

$(BIN): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR) 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN)
