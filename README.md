# YOLOv8 C/C++ Project 

## Layout
- makefile: build/run targets
- prj/: core C++ sources (detection, yolov8, weights)
- test_op/: operator test sources

## Requirements
- g++ with C++11
- Optional: python3 for tooling/scripts (not required for C++ build)

## Quick start
```bash
# Build YOLO C/C++ inference
make build-yolo

# Run YOLO inference
make yolo-detect

# Build operator test binaries
make build-test-operator

# Run selected operator demos/tests
make run-test-operator
```

## Notes
- Data, models, and other large assets are not included;
```bash
git rm -r --cached .
git add -A
```



# YOLOv8n-C-
