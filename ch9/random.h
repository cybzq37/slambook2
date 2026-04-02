#ifndef RAND_H
#define RAND_H

#include <math.h>
#include <stdlib.h>

/**
 * 生成一个0到1之间的随机数
 * @return 随机数
 */
// inline关键字用于建议编译器将该函数在调用处进行内联展开，从而减少函数调用的开销。
inline double RandDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

/**
 * 生成一个标准正态分布的随机数
 * @return 随机数
 */
inline double RandNormal()
{
    double x1, x2, w;
    do{
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    }while( w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w))/w);
    return x1 * w;
}

#endif // random.h