CC = clang
CXX = $(CC)
CFLAGS = -Wall -ggdb -std=c99 -I../common
CFLAGS += `pkg-config opencv --cflags`
LDLIBS = `pkg-config opencv --libs` -lm -lOpenCL

.PHONY: all clean

all: main

main: main.o ../common/libutils.a

clean:
	rm -rf main *.o
