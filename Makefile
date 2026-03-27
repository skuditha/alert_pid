#************************************************************#
#* MAKEFILE for ALERT post-PID C++ extractor                *#
#* Updated for current folder structure                     *#
#************************************************************#

.PHONY: all clean print dirs

# ---- Project layout ----
CPP_DIR   := cpp
SRC_DIR   := $(CPP_DIR)/src
INC_DIR   := $(CPP_DIR)/include
BUILD_DIR := $(CPP_DIR)/build
BIN_DIR   := $(CPP_DIR)/bin

# ---- Final executable ----
TARGET := $(BIN_DIR)/alert_pid_extract

# ---- Tools ----
CXX   := g++
H5CXX := h5c++

# ---- External paths ----
# Expected in environment:
#   HIPO    -> root of hipo4 installation
#   ROOTSYS -> root of ROOT installation

# ---- Includes ----
HIPOCFLAGS   := -I$(HIPO)/include/hipo4
ROOTINCLUDES := -I$(ROOTSYS)/include
USERINCLUDES := -I$(INC_DIR)
LZ4INCLUDES  :=

# ---- Compiler flags ----
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -m64 -pthread
CXXFLAGS += $(USERINCLUDES) $(HIPOCFLAGS) $(LZ4INCLUDES) $(ROOTINCLUDES)

# ---- Linker flags / libs ----
HIPOLIBS := -L$(HIPO)/lib -lhipo4
ROOTLIBS := $(shell root-config --libs 2>/dev/null) $(shell root-config --glibs 2>/dev/null)
LDFLAGS  := -pthread

# ---- Sources / objects ----
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

print:
	@echo "Sources: $(SRCS)"
	@echo "Objects: $(OBJS)"
	@echo "Target:  $(TARGET)"

# ---- Compile ----
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	@echo "Compiling $< ..."
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# ---- Link ----
$(TARGET): $(OBJS)
	@echo "Linking $@ ..."
	$(H5CXX) -O3 -std=c++17 -o $@ $(OBJS) $(LDFLAGS) $(HIPOLIBS) $(ROOTLIBS)

clean:
	@echo "Removing build files..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEPS)