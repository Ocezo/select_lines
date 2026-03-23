# Select lines

`select_lines` is a small C++ program that illustrates binary feature selection based on mutual information and conditional mutual information. It is freely inspired from François Fleuret's paper (2004): "Fast Binary Feature Selection with Conditional Mutual Information".

The program:
- generates `T = 5000` random points in the square `[-1, 1] x [-1, 1]`
- assigns a binary label to each point depending on whether it belongs to a disk of radius `R = 0.5`
- generates `N = 500` random lines, used as binary features
- builds a binary matrix `X` indicating, for each point, on which side of each line it lies
- iteratively selects `K = 15` features by first maximizing mutual information, then updating the scores with conditional mutual information

## What the program shows

At runtime, two OpenCV windows are displayed:
- a view of the 2D points and the candidate lines, then the selected lines
- a view of the score evolution

The score window shows:
- the conditional mutual information of the selected features
- the global mutual information `I(Y; {Xk})`

The program also prints intermediate scores to the console at each selection step.

## Dependencies

The project uses:
- C++17
- Eigen3
- OpenCV
- CMake

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/select_lines
```

Press any key in the OpenCV window at the end to close the application.

## Profiling

A non-interactive mode is available for profiling:

```bash
./build/select_lines --no-gui
```

For reproducible runs, you can also provide a fixed seed:

```bash
./build/select_lines --no-gui --seed=12345
```

To build a profiling-friendly binary with `gprof` support:

```bash
cmake -S . -B build-prof \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSELECT_LINES_ENABLE_PROFILING=ON \
  -DSELECT_LINES_ENABLE_GPROF=ON
cmake --build build-prof
```

Then run:

```bash
rm -f gmon.out
./build-prof/select_lines --no-gui --seed=12345
gprof ./build-prof/select_lines gmon.out
```
