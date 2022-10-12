#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TorchModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)predictAudio:(void*)imageBuffer NS_SWIFT_NAME(predict_conv(image:));
- (nullable NSArray<NSNumber*>*)predictImage:(void*)imageBuffer NS_SWIFT_NAME(predict(image:));
- (nullable NSArray<NSNumber*>*)predictImage_mob:(void*)imageBuffer NS_SWIFT_NAME(predict_mob(image:));

@end

NS_ASSUME_NONNULL_END
