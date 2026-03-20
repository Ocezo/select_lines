# SelectLines

`SelectLines` is a small C++ program that illustrates binary feature selection based on mutual information.

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
./build/SelectLines
```

Press any key in the OpenCV window at the end to close the application.
