PROJECT(NCS_BASE)

FIND_PACKAGE(Protobuf REQUIRED)

FIND_PACKAGE(MPI)
SET(ADDITIONAL_LIBS ${MPI_LIBRARIES})
IF(NCS_CUDA)
  FIND_PACKAGE(CUDA)
  IF (CUDA_FOUND)
    ADD_DEFINITIONS(-DNCS_CUDA)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    IF(NCS_OCELOT)
      SET(ADDITIONAL_LIBS ${ADDITIONAL_LIBS} ocelot)
    ELSE(NCS_OCELOT)
      SET(ADDITIONAL_LIBS ${ADDITIONAL_LIBS} ${CUDA_LIBRARIES})
    ENDIF(NCS_OCELOT)
  ENDIF (CUDA_FOUND)
ENDIF(NCS_CUDA)

INCLUDE_DIRECTORIES("include" ${MPI_INCLUDE_PATH} "${CMAKE_CURRENT_BINARY_DIR}")

#compile CUDA stuff before adding C++11 stuff
IF(CUDA_FOUND AND NCS_CUDA)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  CUDA_COMPILE(cuda_objects
               "source/cuda/MemoryExtractor.cu"
               OPTIONS ${CUDA_OPTIONS} --compiler-options '-fPIC')
  SET(sim_source ${sim_source} ${cuda_objects})
ENDIF(CUDA_FOUND AND NCS_CUDA)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX11_FLAGS}")

FILE(GLOB proto_files "include/ncs/proto/*.proto")
PROTOBUF_GENERATE_CPP(proto_sources proto_headers ${proto_files})

AUX_SOURCE_DIRECTORY("source/spec" spec_source)
ADD_LIBRARY(ncs_spec SHARED ${spec_source} ${proto_headers})
TARGET_LINK_LIBRARIES(ncs_spec ${PROTOBUF_LIBRARIES})
SET_TARGET_PROPERTIES(ncs_spec PROPERTIES LINKER_LANGUAGE CXX)

AUX_SOURCE_DIRECTORY("source/sim" sim_source)

ADD_LIBRARY(ncs_sim SHARED ${sim_source} ${proto_sources})
SET_TARGET_PROPERTIES(ncs_sim PROPERTIES LINKER_LANGUAGE CXX)
TARGET_LINK_LIBRARIES(ncs_sim ncs_spec)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})
FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_SWIG_FLAGS "")
SET_SOURCE_FILES_PROPERTIES("pyncs.i" PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES("pyncs.i" PROPERTIES SWIG_FLAGS "-includeall")
SET(SWIG_MODULE_pyncs_EXTRA_DEPS
    "include/ncs/sim/Simulation.h"
    "include/ncs/spec/AsciiSink.h"
    "include/ncs/spec/DataSource.h"
    "include/ncs/spec/InputGroup.h"
    "include/ncs/spec/BoxGenerator.h"
    "include/ncs/spec/ExactGenerator.h"
    "include/ncs/spec/Generator.h"
    "include/ncs/spec/InputGroup.h"
    "include/ncs/spec/ModelSpecification.h"
    "include/ncs/spec/NormalGenerator.h"
    "include/ncs/spec/NullSink.h"
    "include/ncs/spec/Report.h"
    "include/ncs/spec/UniformGenerator.h"
)
SWIG_ADD_MODULE(pyncs python "pyncs.i")
SWIG_LINK_LIBRARIES(pyncs ${PYTHON_LIBRARIES} ncs_spec ncs_sim)

INSTALL(DIRECTORY include/ncs
        DESTINATION include
		FILES_MATCHING PATTERN "*.h")

IF(UNIX)
	TARGET_LINK_LIBRARIES(ncs_spec pthread dl)
	TARGET_LINK_LIBRARIES(ncs_sim pthread dl ${ADDITIONAL_LIBS})
	INSTALL(TARGETS ncs_spec ncs_sim LIBRARY DESTINATION lib)
ENDIF(UNIX)
