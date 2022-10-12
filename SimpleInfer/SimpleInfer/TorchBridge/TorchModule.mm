#import "TorchModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation TorchModule {
 @protected
  torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer {
  try {
    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, 244, 244}, at::kFloat);
    c10::InferenceMode guard;
    auto outputTensor = _impl.forward({tensor}).toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < 1000; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

- (NSArray<NSNumber*>*)predictAudio:(void*)imageBuffer {
  try {
    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 1, 16000}, at::kFloat);
    c10::InferenceMode guard;
    auto outputTensor = _impl.forward({tensor}).toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < 1000; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

- (NSArray<NSNumber*>*)predictImage_mob:(void*)imageBuffer {
    // 1. the example deeplab.jpg size is size 400x400 and there are 21 semantic classes
    const int WIDTH = 244;
    const int HEIGHT = 244;
    const int CLASSNUM = 21;

    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, WIDTH, HEIGHT}, at::kFloat);
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);

    // 2. convert the input tensor to an NSMutableArray for debugging
    float* floatInput = tensor.data_ptr<float>();
    if (!floatInput) {
        return nil;
    }
    NSMutableArray* inputs = [[NSMutableArray alloc] init];
    for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
        [inputs addObject:@(floatInput[i])];
    }
    auto outputDict = _impl.forward({tensor}).toGenericDict();
    auto outputTensor = outputDict.at("out").toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < CLASSNUM * WIDTH * HEIGHT; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
}

@end

