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

if(OpenCV_FOUND)
  add_executable(ConvertToCVMat ConvertToCVMat.cxx)
  target_link_libraries(ConvertToCVMat gdcmMSFF ${OpenCV_LIBS} glog curl)

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
