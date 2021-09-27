#include <chrono>
#include <iostream>

#include "break_pattern.h"

static const char blox[] = {219, 219};

int main(int argc, char** argv) { 
	auto cool = nbd::BreakPattern(5, 5);
	
	int offset = 0;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      std::cout << (cool.get()[offset++] ? blox : "  ");
		}      
		std::cout << std::endl;
	}
  std::cout << std::endl;

	return 0; 
}