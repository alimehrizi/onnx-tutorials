#ifndef EXTRA_UTILS_H
#define EXTRA_UTILS_H

#include<chrono>



class Timer{
public:
  std::chrono::_V2::steady_clock::time_point start_t;
  Timer();
  void Start();
  double TimeSpent();
  std::chrono::_V2::steady_clock::time_point GetTime();

};

#endif // EXTRA_UTILS_H
