# Makefile for C examples
# Compiles C code to shared libraries for Python ctypes

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC
LDFLAGS = -shared

# Targets
TARGETS = basic_types.so
EXECUTABLES = basic_types

.PHONY: all clean test

all: $(TARGETS) $(EXECUTABLES)

# Shared library for Python ctypes
%.so: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built shared library: $@"

# Standalone executable
%: %.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Built executable: $@"

# Run the standalone executables
test: $(EXECUTABLES)
	@echo "Running C examples..."
	@echo "====================="
	./basic_types

# Clean build artifacts
clean:
	rm -f $(TARGETS) $(EXECUTABLES)
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build all targets (default)"
	@echo "  test       - Run standalone executables"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help message"
