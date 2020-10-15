TARGET = libcerebrum.a
CC=g++
CFLAGS= -Werror -Wall -pedantic -std=c++17 -g  -ggdb -static-libgcc -static-libstdc++
LDFLAGS =
AR = ar
AR_FLAGS = rcs

# Include local paths for libraries
include ../makeconfig


# Check if windows
ifeq ($(OS), Windows_NT)
	CFLAGS := $(CFLAGS) -D WINDOWS_NT

# Not Windows, check if UNIX-based or Linux distro
else
    OS_UNAME := $(shell uname -s)

    ifeq ($(OS_UNAME), Linux)
        CFLAGS := $(CFLAGS) -pthread -D LINUX

    endif

	# Define macros for other UNIX-based systems here

endif




LIBS := $(EIGEN_PATH)



# Include directories for library
SRC_INCLUDE = ./include
LAYERS_INCLUDE := $(SRC_INCLUDE)/Layers
NETWORK_INCLUDE := $(SRC_INCLUDE)/Network

# Source files
SRC_DIR = .
LAYERS_SRC := $(SRC_DIR)/Layers
NETWORK_SRC := $(SRC_DIR)/Network


# Object files
NETWORK_OBJS = Network.o Cost.o
LAYER_OBJS = Layer.o FullyConnected.o Activations.o

OBJECTS := $(LAYER_OBJS) $(NETWORK_OBJS)

INCLUDE_DIRECTORIES = $(NETWORK_INCLUDE) $(LIBS) $(LAYERS_INCLUDE)
INCLUDE_DIRECTORIES := $(addprefix -I, $(INCLUDE_DIRECTORIES))


.PHONY: all clean cleanrm





# Rules
all: $(TARGET)


$(TARGET): $(OBJECTS)
	$(AR) $(AR_FLAGS) $@ $(OBJECTS)


Layer.o: $(LAYERS_SRC)/Layer.cpp $(LAYERS_INCLUDE)/Layer.h
	$(CC) -c $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRECTORIES) $<


FullyConnected.o: $(LAYERS_SRC)/FullyConnected.cpp $(LAYERS_INCLUDE)/FullyConnected.h
	$(CC) -c $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRECTORIES) $<


Network.o: $(NETWORK_SRC)/Network.cpp $(NETWORK_INCLUDE)/Network.h
	$(CC) -c $(CFLAGS)$(LDFLAGS) $(INCLUDE_DIRECTORIES) $<


Activations.o: $(LAYERS_SRC)/Activations.cpp $(LAYERS_INCLUDE)/Activations.h
	$(CC) -c $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRECTORIES) $<


Cost.o: $(NETWORK_SRC)/Cost.cpp $(NETWORK_INCLUDE)/Cost.h
	$(CC) -c $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRECTORIES) $<



#for windows cmd
clean:
	del  /Q /S *.o $(TARGET)

#for cmderr unix-based commands
cleanrm:
	rm *.o $(TARGET)
