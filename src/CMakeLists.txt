# Recurrent RNN Models
#include_directories("${OnlineRNN_SOURCE_DIR}/include/CustomModels/")
add_library(RecurrentNNTorch OBJECT CustomModels/RecurrentNNTorch.cpp)
target_include_directories(
        RecurrentNNTorch PUBLIC ${OnlineRNN_SOURCE_DIR}/include/CustomModels/)
target_link_libraries(RecurrentNNTorch "${TORCH_LIBRARIES}")
# Multithreading containers
add_library(
        MultithreadContainers OBJECT MultithreadContainers/StringHolder.cpp
        ${OnlineRNN_SOURCE_DIR}/include/MultithreadContainers/MultithreadQueue.h
        ${OnlineRNN_SOURCE_DIR}/include/constants.h ../include/MultithreadContainers/WeightHolder.h)
target_include_directories(
        MultithreadContainers PRIVATE ${OnlineRNN_SOURCE_DIR}/include/MultithreadContainers/
        PRIVATE ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(MultithreadContainers PUBLIC ${TORCH_LIBRARIES})
# Models
add_library(myModels OBJECT ${OnlineRNN_SOURCE_DIR}/src/Models/TorchRNN.cpp)

target_include_directories(myModels PRIVATE ${OnlineRNN_SOURCE_DIR}/include/ PRIVATE C:/Users/zhymi/cpp_work/OnlineRNN/include/CustomModels)

target_link_libraries(myModels ${TORCH_LIBRARIES} OBJECT $<TARGET_OBJECTS:recurrentNNTorch>)