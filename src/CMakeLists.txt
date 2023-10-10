find_package(Torch REQUIRED)
# forecaster
add_library(forecaster forecaster.cpp)
target_include_directories(forecaster PRIVATE ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(forecaster ${TORCH_LIBRARIES} recurrentNNTorch MultithreadContainers)


# Recurrent RNN Models
add_library(recurrentNNTorch CustomModels/RecurrentNNTorch.cpp)
target_include_directories(
        recurrentNNTorch PUBLIC ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(recurrentNNTorch PRIVATE ${TORCH_LIBRARIES})

# Multithreading containers
add_library(
        MultithreadContainers MultithreadContainers/StringHolder.cpp
        ${OnlineRNN_SOURCE_DIR}/include/MultithreadContainers/MultithreadQueue.h
        ${OnlineRNN_SOURCE_DIR}/include/constants.h MultithreadContainers/WeightHolder.cpp)
target_include_directories(
        MultithreadContainers PUBLIC ${OnlineRNN_SOURCE_DIR}/include/MultithreadContainers/
        PUBLIC ${OnlineRNN_SOURCE_DIR}/include/)
target_link_libraries(MultithreadContainers PRIVATE ${TORCH_LIBRARIES})

# Models
add_library(myModels Models/TorchRNN.cpp Models/TorchMLP.cpp)
target_include_directories(myModels PUBLIC ${OnlineRNN_SOURCE_DIR}/include/CustomModels/ PUBLIC ${OnlineRNN_SOURCE_DIR}/include)
target_link_libraries(myModels PRIVATE ${TORCH_LIBRARIES} PUBLIC recurrentNNTorch)