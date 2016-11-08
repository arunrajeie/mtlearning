CXX = g++
#CXX = g++-5

%CXXFLAGS = -O3 -g -std=c++11 -DNDEBUG -Wno-deprecated -pedantic -ffast-math -I./ 
CXXFLAGS = -O3 -g -DNDEBUG -D_PARALLEL_ -fopenmp -Wno-deprecated -pedantic -ffast-math -I./ 
LINKFLAGS = -lm

SRCS=\
	MLearning.cpp \
	lbfgs.cpp 
SRCS1= \
	Train.cpp 
SRCS2= \
	Predict.cpp 
SRCS3= \
	ExtractFeature.cpp 

OBJS  = $(SRCS:%.cpp=%.o)
OBJS1 = $(SRCS1:%.cpp=%.o)
OBJS2 = $(SRCS2:%.cpp=%.o)
OBJS3 = $(SRCS3:%.cpp=%.o)

all: train predict extract
train: $(OBJS) $(OBJS1)
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) $(OBJS) $(OBJS1) $(LINKFLAGS) -o mlearning-train
predict: $(OBJS) $(OBJS2)
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) $(OBJS) $(OBJS2) $(LINKFLAGS) -o mlearning-predict
extract: $(OBJS) $(OBJS3)
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) $(OBJS) $(OBJS3) $(LINKFLAGS) -o mlearning-extract
debug:
	make all CXXFLAGS="-ggdb -W -Wall -pedantic"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) -c $<

clean:
	rm -f mlearning-train mlearning-predict mlearning-extract *.o *.lo *~
