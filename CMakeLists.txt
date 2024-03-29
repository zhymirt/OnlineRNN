cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#
#add_executable(example-app example-app.cpp example-app.h)
#target_link_libraries(example-app "${TORCH_LIBRARIES}")
#set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
#cmake_minimum_required(VERSION 3.23)
project(OnlineRNN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# online_rnn
add_executable(online_rnn src/main.cpp)
#add_library(OnlineRNN library.cpp forecaster.cpp forecaster.h "${TORCH_LIBRARIES}")
add_subdirectory(src)
target_include_directories(online_rnn PUBLIC ${OnlineRNN_SOURCE_DIR}/include/ ${OnlineRNN_SOURCE_DIR}/include/MultithreadContainers/)
target_link_directories(online_rnn PRIVATE ${OnlineRNN_SOURCE_DIR}/src)
target_link_libraries(online_rnn "${TORCH_LIBRARIES}" forecaster recurrentNNTorch MultithreadContainers)

# test_online_rnn
add_executable(test_online_rnn tests/test_main.cpp)
add_subdirectory(tests)
target_include_directories(test_online_rnn PRIVATE ${OnlineRNN_SOURCE_DIR}/include/ ${OnlineRNN_SOURCE_DIR}/src/Models/)
target_link_directories(test_online_rnn PRIVATE ${OnlineRNN_SOURCE_DIR}/tests)
target_link_libraries(test_online_rnn ${TORCH_LIBRARIES} test_forecaster testMultiThreadContainers myModels test_torchRNN)


if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET online_rnn
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:online_rnn>)
endif (MSVC)