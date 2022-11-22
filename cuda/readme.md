# Build

```sh
mkdir build
cd build

# on windows
cmake .. -A x64 -DCMAKE_CUDA_COMPILER:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe" 
cmake --build . --config Release
```

# Run
```sh
./build/Release/main.exe cublas 1024 1024 1024
```