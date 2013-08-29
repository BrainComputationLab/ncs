#pragma once
#include <string>
#include <vector>

namespace ncs {

namespace sim {

class File {
public:
  static bool isDirectory(const std::string& path);
  static bool isFile(const std::string& path);
  static std::vector<std::string> getContents(const std::string& path);
  static std::vector<std::string> getAllFiles(const std::string& path);
private:
};

} // namespace sim

} // namespace ncs
