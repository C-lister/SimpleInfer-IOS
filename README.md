### Requirements

- XCode 11.0 or above
- iOS 12.0 or above

## SimpleInfer

### File Usage

- save_pt.py: An example to create a **.pt** script for mobile
- Model: Store five model scripts in **.pt** format
- TorchBridge: Use Object-C to load torch scripts, create tensors and do inference
- ViewController.swift: Initialize screen and show inference time
- UIImage+Helper.swift: Convert image to float arrays

### Install LibTorch via Cocoapods

The PyTorch C++ library is available in [Cocoapods](https://cocoapods.org/), to integrate it to our project, we can run

```ruby
pod install
```
Then, open the project in Xcode

```
open SimpleInfer.xcworkspace
```

