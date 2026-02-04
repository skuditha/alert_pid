#************************************************************#
#* MAKEFILE for compiling C++ files to read HIPO4 into ROOT	*#
#* AUTHOR: SKUDITHA - July 22, 2024							*#
#* STEPS to build and run a C++ code named filename.cc:		*#
#*   (1) make clean											*#
#*   (2) make filename.o									*#
#*   (3) make filename										*#
#*   (4) ./filename											*#
#************************************************************#

.PHONY = all clean print

HIPOCFLAGS  := -I$(HIPO)/include/hipo4      
HIPOLIBS    := -L$(HIPO)/lib

LZ4LIBS     := -L$(ROOTSYS)/lib/ -lhipo4 -lCore -lCore -lCling -lRIO -lHist -lTree -lGpad -lGraf -lRint -lGraf3d -lPhysics -lMathCore -lMatrix -lTreePlayer -lGX11TTF -lGX11 -lGui -lGed -lGeom -lFitPanel
LZ4INCLUDES := -I$(ROOTSYS)/lib/include

ROOTLIBS     := -L$(ROOTSYS)/lib -lCore -lCling -lRIO -lHist -lTree -lGpad -lGraf -lRint -lGraf3d -lPhysics -lMathCore -lMatrix -lTreePlayer -lGX11TTF -lGX11 -lGui -lGed -lGeom -lFitPanel
ROOTINCLUDES := -I$(ROOTSYS)/include

CXX       := g++
CXXFLAGS  += -std=c++17 -Wall -Wextra -m64 -pthread -rdynamic  `root-config --libs` `root-config --glibs` 
LD        := g++ 
LDFLAGS   += -pthread

SRCS := $(wildcard *.cc)
BINS := $(SRCS:%.cc=%)

all: ${BINS}

print:
	@echo "Making..."
	@echo ${SRCS}

$(BINS): %: %.o
	@echo "Checking..."
	$(CXX) -o $@ $< $(HIPOLIBS) $(LZ4LIBS) 

clean:
	@echo 'Removing all build files...'
	@rm -rf *.o $(BINS)

%.o: %.cc
	@echo "Creating object..."
	$(CXX) -c $< -O3 $(CXXFLAGS) $(HIPOCFLAGS) $(LZ4INCLUDES) $(ROOTINCLUDES)
