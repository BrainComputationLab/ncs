#include <dirent.h>
#include <iostream>
#include <queue>
#include <stddef.h>
#include <sys/stat.h>

#include <ncs/sim/File.h>

namespace ncs {

namespace sim {

bool File::isDirectory(const std::string& path) {
  struct stat stats;
  int result = stat(path.c_str(), &stats);
  return S_ISDIR(stats.st_mode);
}

bool File::isFile(const std::string& path) {
  struct stat stats;
  int result = stat(path.c_str(), &stats);
  return S_ISREG(stats.st_mode);
}

std::vector<std::string> File::getContents(const std::string& path) {
  std::vector<std::string> contents;
  DIR* directory = opendir(path.c_str());
  if (nullptr == directory) {
    std::cerr << "Could not open directory " << path << std::endl;
    return contents;
  }
  long max_path_length = fpathconf(dirfd(directory), _PC_NAME_MAX);
  if (max_path_length < 0) {
    std::cerr << "Could not determine maximum path length" << std::endl;
    std::terminate();
  }
  long dirent_length = offsetof(struct dirent, d_name) + max_path_length + 1;
  struct dirent* entry = (struct dirent*)malloc(dirent_length);
  struct dirent* result = entry;
  while (0 == readdir_r(directory, entry, &result) && nullptr != result) {
    if (entry->d_type == DT_REG || entry->d_type == DT_REG) {
      contents.push_back(path + "/" + std::string(entry->d_name));
    }
  }
  free(entry);
  return contents;
}

std::vector<std::string> File::getAllFiles(const std::string& path) {
  std::vector<std::string> files;
  std::queue<std::string> paths_to_check;
  paths_to_check.push(path);
  while (!paths_to_check.empty()) {
    std::string current_path = paths_to_check.front();
    paths_to_check.pop();
    if (isFile(current_path)) {
      files.push_back(current_path);
    }
    else if (isDirectory(current_path)) {
      std::vector<std::string> contents = getContents(current_path);
      for (const auto& c : contents) {
        paths_to_check.push(c);
      }
    }
  }
  return files;
}

} // namespace sim

} // namespace ncs
