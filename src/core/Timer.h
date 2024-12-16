#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>

class Timer
{
public:
    Timer(std::string label);
    ~Timer();

private:
    std::string label;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#endif // TIMER_H