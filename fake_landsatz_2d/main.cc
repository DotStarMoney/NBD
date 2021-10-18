#include <stdint.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "synthesize.h"

void WriteTimeSeries(const nbd::SynthesizeConfig& config, 
                     const nbd::SynthesizeResult& result,
                     std::ofstream* output_file) {
  int elements_n = config.depth * config.height * config.width;
  output_file->write(reinterpret_cast<char*>(result.raw.get()),
                     sizeof(float) * elements_n);
  output_file->write(reinterpret_cast<char*>(result.spine.get()),
                     sizeof(float) * elements_n);
  output_file->write(reinterpret_cast<char*>(result.distance.get()),
                     sizeof(float) * elements_n);
  output_file->write(reinterpret_cast<char*>(result.mask.get()),
                     sizeof(uint8_t) * elements_n);
}

int main(int argc, char** argv) { 
  std::cout << "Start!" << std::endl;

  std::ofstream output_file("test.dat", std::ios::out | std::ios::binary);
  int records_n = 10;

  output_file.write(reinterpret_cast<char*>(&records_n), sizeof(int));

  for (int i = 0; i < records_n; ++i) {
    auto ts = nbd::Synthesize();
    WriteTimeSeries(nbd::kDefaultConfig, ts, &output_file);
    std::cout << i << std::endl;
  }

  std::cout << "Stop!" << std::endl;

  return 0; 
}
