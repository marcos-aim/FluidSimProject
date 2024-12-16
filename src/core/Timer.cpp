#include <timer.h>

#include <iostream>
#include <utility>

Timer::Timer(std::string label)
    : label(std::move(label))
{
    start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    auto end = std::chrono::high_resolution_clock::now();

    auto sTime = std::chrono::time_point_cast<std::chrono::microseconds>(start)
        .time_since_epoch()
        .count();
    auto eTime = std::chrono::time_point_cast<std::chrono::microseconds>(end)
        .time_since_epoch()
        .count();

    auto duration = (eTime - sTime) * 0.001;
    std::cerr << "<" << label << "> clock ended after: " << duration << std::endl;
}