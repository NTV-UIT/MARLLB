# Makefile for Reservoir Sampling Implementation
# 
# Targets:
#   all         - Build C implementation
#   test        - Run Python tests
#   example     - Run Python examples
#   benchmark   - Run C benchmark
#   clean       - Remove build artifacts
#
# Author: MARLLB Implementation Team
# Date: December 2025

CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -march=native
LDFLAGS = -lm

SRC_DIR = src
TEST_DIR = tests
EXAMPLE_DIR = examples
BUILD_DIR = build

# Source files
C_SOURCES = $(SRC_DIR)/reservoir.c
C_HEADERS = $(SRC_DIR)/reservoir.h
PY_SOURCES = $(SRC_DIR)/reservoir.py $(SRC_DIR)/features.py

# Executables
C_EXECUTABLE = $(BUILD_DIR)/reservoir_demo

# Python interpreter
PYTHON = python3

.PHONY: all test example benchmark clean help

# Default target
all: $(C_EXECUTABLE)
	@echo "Build complete!"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build C implementation
$(C_EXECUTABLE): $(C_SOURCES) $(C_HEADERS) | $(BUILD_DIR)
	@echo "Compiling C implementation..."
	$(CC) $(CFLAGS) $(C_SOURCES) -o $(C_EXECUTABLE) $(LDFLAGS)
	@echo "C executable: $(C_EXECUTABLE)"

# Run Python unit tests
test:
	@echo "Running Python unit tests..."
	$(PYTHON) $(TEST_DIR)/test_reservoir.py

# Run Python examples
example:
	@echo "Running Python examples..."
	$(PYTHON) $(EXAMPLE_DIR)/basic_usage.py

# Run C benchmark
benchmark: $(C_EXECUTABLE)
	@echo "Running C benchmark..."
	./$(C_EXECUTABLE)

# Run C example
run-c: $(C_EXECUTABLE)
	@echo "Running C demo..."
	./$(C_EXECUTABLE)

# Check code style
style:
	@echo "Checking Python code style..."
	-$(PYTHON) -m pylint $(SRC_DIR)/*.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf $(EXAMPLE_DIR)/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Clean complete!"

# Show help
help:
	@echo "MARLLB Reservoir Sampling - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build C implementation"
	@echo "  test        - Run Python unit tests"
	@echo "  example     - Run Python examples"
	@echo "  benchmark   - Run C performance benchmark"
	@echo "  run-c       - Run C demo"
	@echo "  style       - Check Python code style"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make all test example    # Build and test everything"
	@echo "  make benchmark           # Run performance benchmark"
