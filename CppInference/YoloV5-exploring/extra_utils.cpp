#include "extra_utils.h"

Timer::Timer(){

}
void Timer::Start(){
    start_t = std::chrono::steady_clock::now();

}
double Timer::TimeSpent(){
    auto end_t = std::chrono::steady_clock::now();
    double t_ = ( end_t - start_t).count()/1e6;
    return t_;
}
std::chrono::_V2::steady_clock::time_point Timer::GetTime(){
    auto t = std::chrono::steady_clock::now();
    return t;
}
