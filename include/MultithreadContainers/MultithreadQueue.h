//
// Created by zhymi on 5/25/2023.
//

#ifndef MULTITHREADCONTAINERS_MULTITHREADQUEUE_H
#define MULTITHREADCONTAINERS_MULTITHREADQUEUE_H

#include <mutex>
#include <queue>

template <typename T> class MultithreadQueue {
private:
    std::mutex queueMutex;
    std::queue<T> queue;
public:
    MultithreadQueue() {
        this->queue = std::queue<T>();
    }
    void push(T item) {
        std::lock_guard<std::mutex> lock(this->queueMutex);
        this->queue.push(item);
    }
    void pop() { // should check before doing a random pull
        std::lock_guard<std::mutex> lock(this->queueMutex);
        this->queue.pop();
    }
    T front() {
        std::lock_guard<std::mutex> lock(this->queueMutex);
        return this->queue.front();
    }
    T back() {
        std::lock_guard<std::mutex> lock(this->queueMutex);
        return this->queue.back();
    }
    bool empty() {
        std::lock_guard<std::mutex> lock(this->queueMutex);
        return this->queue.empty();
    }
    ~MultithreadQueue() {  // finish the destructor
//        std::lock_guard<std::mutex> unlock;
        delete &this->queueMutex;
        this->queue = {};
    }
};
//#include "MultithreadQueue.ipp"

#endif // MULTITHREADCONTAINERS_MULTITHREADQUEUE_H
