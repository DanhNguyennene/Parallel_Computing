CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall -Wextra -march=native
LDFLAGS = -fopenmp
TARGET = main
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) main *.txt *.csv

test: $(TARGET)
	mpirun -np 7 -x OMP_NUM_THREADS=7 ./$(TARGET) 100 1 7 128

test-sizes: $(TARGET)
	mpirun -np 7 -x OMP_NUM_THREADS=7 ./$(TARGET) 100 1 7 128
	mpirun -np 7 -x OMP_NUM_THREADS=7 ./$(TARGET) 1000 1 7 128
	mpirun -np 7 -x OMP_NUM_THREADS=7 ./$(TARGET) 4000 0 7 128

test-threads: $(TARGET)
	mpirun -np 7 -x OMP_NUM_THREADS=1 ./$(TARGET) 1000 0 1 128
	mpirun -np 7 -x OMP_NUM_THREADS=2 ./$(TARGET) 1000 0 2 128
	mpirun -np 7 -x OMP_NUM_THREADS=4 ./$(TARGET) 1000 0 4 128
	mpirun -np 7 -x OMP_NUM_THREADS=7 ./$(TARGET) 1000 0 7 128

.PHONY: all clean test test-sizes test-threads
	@echo "Clean complete"
