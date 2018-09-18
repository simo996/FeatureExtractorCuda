
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
    /**
     * Create a folder with the specified path/name
     * @param folderPath
     */
    static void createFolder(string folderPath);
    /**
     * Removes the path symbols to a file an returns only its name
     * @param pathname
     * @return
     */
    static string basename(string const& pathname);
    /**
     * Removes the extension of a file
     * @param filename
     * @return
     */
    static string removeExtension( std::string const& filename );
};


#endif //PRE_CUDA_UTILS_H
