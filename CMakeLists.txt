cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#project(OnlineRNN)
#
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
#add_executable(example-app example-app.cpp example-app.h)
#target_link_libraries(example-app "${TORCH_LIBRARIES}")
#set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
#cmake_minimum_required(VERSION 3.23)
project(OnlineRNN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
add_executable(online_rnn forecaster.cpp forecaster.h)
#add_library(OnlineRNN library.cpp forecaster.cpp forecaster.h "${TORCH_LIBRARIES}")
target_link_libraries(online_rnn "${TORCH_LIBRARIES}")
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET online_rnn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:online_rnn>)
endif (MSVC)