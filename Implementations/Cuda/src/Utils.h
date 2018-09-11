
#ifndef PRE_CUDA_UTILS_H
#define PRE_CUDA_UTILS_H

#include <string>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sys/stat.h> // file system interaction


using namespace std;

// UNIX
struct MatchPathSeparator
{
    bool operator()( char ch ) const
    {
        return ch == '/';
    }
};

class Utils {
public:
    // File system interaction
    static void createFolder(string folderPath);
    static string basename(string const& pathname);
    static string removeExtension( std::string const& filename );
};


#endif //PRE_CUDA_UTILS_H
