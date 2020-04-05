#include <iostream>
using namespace std;
//// 1. hello C++
//int main() {
//    cout << "Hello!" << endl;
//    cout << "Welcome to C++!" << endl;
//    return 0;
//}

//// 2. cin
//int main() {
//    const double pi = 3.14159;
//    int radius = 0;
//
//    cout << "please enter the radius!\n";
//    cin >> radius;
//    cout << "The radius is:" << radius << '\n';
//    cout << "PI is:" << pi << '\n';
//    cout << "please aenter a different radius!\n";
//    cin >> radius;
//    cout << "Now the radius is changed to:" << radius << '\n';
//    cout << "PI is still:" << pi << '\n';
//    return 0;
//}

// 3. expression
int main() {
    int a = 0;
    int b = 0;
    int x = 0;
    x = (a-b) > 0 ? (a - b) : (b - a);
    cout << "The value of x is" << x;
    return 0
}