`#pragma unroll`:

c++中的一种编译优化操作，用于将循环操作进行展开表示，以加快编译。

```cpp
#pragma unroll
for ( int i = 0; i < 5; i++ )
    b[i] = i;

在编译时展开为
b[0] = 0;
b[1] = 1;
b[2] = 2;
b[3] = 3;
b[4] = 4;
```