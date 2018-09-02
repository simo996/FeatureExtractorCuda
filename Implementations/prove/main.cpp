#include <iostream>
#include <vector>

using namespace std;

struct initalpointer()
{
    int *pixels;
};

struct threadpointers(){
    int* actualpxs;
};

void test(int * elements, int lenght, int cols){

    for (int i = 0; i < lenght; ++i) {
        int *actualElements = elements + (i * cols);
        for (int j = 0; j < cols; ++j) {
            cout << actualElements[j] << " ";
        }
        cout << endl;
    }
}
int main() {
    vector<int> pixels = {10, 2, 4, 6,
                          7, 7, 134, 21,
                          2, 4, 6, 13, 52};

    int* pixelsPointer = pixels.data();
    int nrows= 3;
    int cols = 4;


    test(pixelsPointer, 2, cols);

    return 0;
}