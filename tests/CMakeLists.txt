find_package(Torch REQUIRED)

add_library(test_forecaster test_forecaster.cpp)
target_include_directories(test_forecaster PRIVATE ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(test_forecaster forecaster myModels)

add_library(test_torchRNN Models/testTorchRNN.cpp)
target_include_directories(test_torchRNN PRIVATE ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(test_torchRNN ${TORCH_LIBRARIES} myModels)

add_library(testMultiThreadContainers MultithreadContainers/test_weightHolder.cpp MultithreadContainers/test_multithreadQueue.cpp MultithreadContainers/test_multithreadQueue.h)
target_include_directories(testMultiThreadContainers PUBLIC {OnlineRNN_SOURCE_DIR}/include/ PUBLIC MultithreadContainers/)
target_link_libraries(testMultiThreadContainers ${TORCH_LIBRARIES} MultithreadContainers)

