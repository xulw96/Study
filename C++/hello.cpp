#include <stdio.h>

class Rectangle {
    int * width;
    int * height;
public:
    Rectangle(int, int);
    void printMe(){
        printf("Dimensions: %d by %d.\n", *width, *height);
    }
};
Rectangle::Rectangle(int w, int h){
    width = new int;
    height = new int;
    *width = w;
    *height = h;
}
int main(){
    Rectangle box(5, 7);
    box.printMe();
}