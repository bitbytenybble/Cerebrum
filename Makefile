# Include local paths for libraries
include makeconfig

TARGET = cerebrum
CC=g++
CFLAGS = -Werror -Wall -pedantic -std=c++17 -g  -ggdb  -static-libgcc -static-libstdc++ -pthread
LDFLAGS = -L./src -lcerebrum

# Libraries to include
LIBS := $(EIGEN_PATH) $(PLOT_PATH)


# Check which operating system this Makefile is running on



# Include directories for library
SRC_INCLUDE = src/include
LAYERS_INCLUDE := $(SRC_INCLUDE)/Layers
NETWORK_INCLUDE := $(SRC_INCLUDE)/Network


# Test directories
TEST_INCLUDE = test
TEST_OBJS = test_utils.o





# Object files
OBJECTS := $(TEST_OBJS)

INCLUDE_DIRECTORIES := $(NETWORK_INCLUDE) $(LIBS) $(LAYERS_INCLUDE)
INCLUDE_DIRECTORIES := $(addprefix -I, $(INCLUDE_DIRECTORIES))



.PHONY: all clean cleanrm cleanAll cleanAllrm cerebrum_lib



# Rules
all: $(TARGET)




$(TARGET): $(OBJECTS) cerebrum_lib
	$(CC) $(CFLAGS) $(INCLUDE_DIRECTORIES) $< main.cpp $(LDFLAGS) -o $(TARGET)


test_utils.o: $(TEST_INCLUDE)/test_utils.cpp $(TEST_INCLUDE)/test_utils.h
	$(CC) -c $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRECTORIES) $<




cerebrum_lib:
	cd src && $(MAKE) all


#for windows cmd
clean:
	del  /Q /S *.o *.exe *.csv

#for cmderr unix-based commands
cleanrm:
	rm ./*.o ./*.exe ./*.csv $(TARGET)*



cleanAllrm:
	cd src && $(MAKE) cleanrm
	$(MAKE) cleanrm


cleanAll:
	cd src && $(MAKE) clean
	$(MAKE) clean
