#include <chrono>
#include <iostream>

#include "synthesize.h"

int main(int argc, char** argv) { 
  std::cout << "Start!";

  auto cool = nbd::Synthesize();

  std::cout << "Stop!";

  return 0; 
}
