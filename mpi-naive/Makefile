CXX = mpicxx
CXXFLAGS = -O2 -std=c++11
TARGET = main
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
N ?= 1000
NP ?= 8

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

run: $(TARGET)
	mpirun -np $(NP) --use-hwthread-cpus ./$(TARGET) $(N)

clean:
	rm -f $(TARGET)