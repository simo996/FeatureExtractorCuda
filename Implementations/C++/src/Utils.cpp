
#include "Utils.h"

/* Support code for putting the results in the right output folder */
void Utils::createFolder(string folderPath){
    if (mkdir(folderPath.c_str(), 0777) == -1) {
        if (errno == EEXIST) {
            // alredy exists
        } else {
            // something else
            cerr << "cannot create save folder: " << folderPath << endl
                 << "error:" << strerror(errno) << endl;
        }
    }
}


// remove the path and keep filename+extension
string Utils::basename( std::string const& pathname ){
    return string(
            find_if( pathname.rbegin(), pathname.rend(),
                     MatchPathSeparator() ).base(),
            pathname.end() );
}

// remove extension from filename
string Utils::removeExtension( std::string const& filename ){
    string::const_reverse_iterator
            pivot
            = find( filename.rbegin(), filename.rend(), '.' );
    return pivot == filename.rend()
           ? filename
           : std::string( filename.begin(), pivot.base() - 1 );
}