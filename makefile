CXX := g++
CXXFLAGS := -IC:/w64devkit/include/include -std=c++17 -Ofast -march=native -flto -mavx2 -mfma -fopenmp -Wall  -g
LDFLAGS := -fopenmp
SRC_DIR := src
BUILD_DIR := build
BIN := cppDL

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
