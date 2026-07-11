/**
 * basic_types.c
 *
 * Demonstrates C memory basics: primitive types, structs, and memory layout.
 * Compile with: gcc -shared -fPIC -o basic_types.so basic_types.c
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Simple struct demonstrating padding */
struct Point {
    int32_t x;
    int32_t y;
};

/* Struct with padding issues */
struct UnalignedStruct {
    char a;      /* 1 byte */
    int32_t b;   /* 4 bytes */
    char c;      /* 1 byte */
    /* Total with padding: 12 bytes */
};

/* Optimized struct (fields reordered) */
struct OptimizedStruct {
    int32_t b;   /* 4 bytes */
    char a;      /* 1 byte */
    char c;      /* 1 byte */
    /* Total with padding: 8 bytes */
};

/* Function to demonstrate integer sizes */
void print_type_sizes(void) {
    printf("Integer Type Sizes:\n");
    printf("==================\n");
    printf("char:           %zu bytes\n", sizeof(char));
    printf("short:          %zu bytes\n", sizeof(short));
    printf("int:            %zu bytes\n", sizeof(int));
    printf("long:           %zu bytes\n", sizeof(long));
    printf("long long:      %zu bytes\n", sizeof(long long));
    printf("float:          %zu bytes\n", sizeof(float));
    printf("double:         %zu bytes\n", sizeof(double));
    printf("pointer:        %zu bytes\n", sizeof(void*));
    printf("\n");
}

/* Function to demonstrate struct sizes and padding */
void print_struct_sizes(void) {
    printf("Struct Sizes:\n");
    printf("=============\n");
    printf("Point:            %zu bytes\n", sizeof(struct Point));
    printf("UnalignedStruct:  %zu bytes\n", sizeof(struct UnalignedStruct));
    printf("OptimizedStruct:  %zu bytes\n", sizeof(struct OptimizedStruct));
    printf("\n");
}

/* Function to demonstrate stack allocation */
int32_t stack_example(void) {
    int32_t x = 42;  /* Stack allocated */
    int32_t y = 100;
    return x + y;
}

/* Function to demonstrate heap allocation */
int32_t* heap_example(void) {
    int32_t* p = (int32_t*)malloc(sizeof(int32_t));
    if (p != NULL) {
        *p = 42;
    }
    return p;  /* Caller must free! */
}

/* Function to demonstrate integer overflow (undefined behavior in C) */
int32_t overflow_example(int32_t value) {
    return value + 1;  /* Can overflow if value is INT32_MAX */
}

/* Function to compute factorial (shows recursive stack usage) */
uint64_t factorial(uint64_t n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

/* Main function for standalone testing */
int main(void) {
    printf("C Memory Basics Demo\n");
    printf("====================\n\n");

    print_type_sizes();
    print_struct_sizes();

    /* Stack example */
    printf("Stack Example:\n");
    printf("Result: %d\n\n", stack_example());

    /* Heap example */
    printf("Heap Example:\n");
    int32_t* heap_ptr = heap_example();
    if (heap_ptr != NULL) {
        printf("Heap value: %d\n", *heap_ptr);
        free(heap_ptr);  /* Must free! */
    }
    printf("\n");

    /* Overflow example */
    printf("Overflow Example:\n");
    int32_t max_int = 2147483647;  /* INT32_MAX */
    printf("INT32_MAX: %d\n", max_int);
    printf("INT32_MAX + 1: %d (overflow!)\n", overflow_example(max_int));
    printf("\n");

    /* Factorial example */
    printf("Factorial Example:\n");
    printf("5! = %llu\n", factorial(5));
    printf("20! = %llu\n", factorial(20));

    return 0;
}
