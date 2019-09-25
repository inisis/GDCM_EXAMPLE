# Build the GDCM examples
# Add the include paths
include_directories(
  "${GDCM_BINARY_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/Common"
  "${GDCM_SOURCE_DIR}/Source/DataStructureAndEncodingDefinition"
  "${GDCM_SOURCE_DIR}/Source/MediaStorageAndFileFormat"
  "${GDCM_SOURCE_DIR}/Source/InformationObjectDefinition"
  "${GDCM_SOURCE_DIR}/Source/MessageExchangeDefinition"

# FIXME:
  "${GDCM_SOURCE_DIR}/Source/DataDictionary"
  "${GDCM_SOURCE_DIR}/Utilities"
  )
# error C1083: Cannot open include file: 'zconf.h'
if(NOT GDCM_USE_SYSTEM_ZLIB)
include_directories(
  "${GDCM_BINARY_DIR}/Utilities/gdcmzlib"
)
endif()

find_package(OpenCV REQUIRED)

if(GDCM_USE_JPEGLS)
set(EXAMPLES_SRCS
  ${EXAMPLES_SRCS}
  FixJAIBugJPEGLS
  )
endif()

SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
find_package(CUDA REQUIRED)
# /usr/share/cmake/Modules/FindCUDA.cmake
# INCLUDE(FindCUDA)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the CUDAConfig.cmake file.
# Print some message showing some of them
message(STATUS "CUDA library status:")
message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")


set(TensorRT_PATH "/home/yaojin/production/TensorRT-5.1.5.0")
include_directories(${TensorRT_PATH}/include ${CUDA_INCLUDE_DIRS})
link_directories("${TensorRT_PATH}/lib")

if(OpenCV_FOUND)
  add_executable(ConvertToCVMat ConvertToCVMat.cxx)
  target_link_libraries(ConvertToCVMat gdcmMSFF ${OpenCV_LIBS} glog curl)

  add_executable(ConvertToCVMat_PERF ConvertToCVMat_PERF.cxx)
  target_link_libraries(ConvertToCVMat_PERF gdcmMSFF ${OpenCV_LIBS} glog curl)

  add_executable(ConvertToCVMat_TensorRT ConvertToCVMat_TensorRT.cxx)
  target_link_libraries(ConvertToCVMat_TensorRT gdcmMSFF ${OpenCV_LIBS} nvinfer_plugin nvcaffe_parser nvinfer ${CUDA_LIBRARIES} glog curl)
endif()

if(BUILD_SHARED_LIBS)
  set_source_files_properties(FixJAIBugJPEGLS.cxx
                              PROPERTIES
                              COMPILE_FLAGS -DCHARLS_DLL
                              )
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND MINGW)
  set_target_properties( ReadExplicitLengthSQIVR PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition")
endif()
