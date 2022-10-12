import UIKit

class ViewController: UIViewController {
    @IBOutlet var resultView: UITextView!

    override func viewDidLoad() {
        super.viewDidLoad()
        let image = UIImage(named: "image.png")!
        let resizedImage = image.resized(to: CGSize(width: 244, height: 244))
        var text = ""
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        guard var audioBuffer = resizedImage.audio() else {
            return
        }
        //inception
        guard let filePath = Bundle.main.path(forResource: "inception_v3", ofType: "pt"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        let startDate_inc = Date()
        for _ in 0 ..< 10 {
            guard let output_inc = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
                return
            }
        }
        let inferenceTime_inc = (Date().timeIntervalSince(startDate_inc) * 1000)/10
        //resnet
        guard let filePath = Bundle.main.path(forResource: "resnet_101", ofType: "pt"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        let startDate_res = Date()
        for _ in 0 ..< 10 {
            guard let output_res = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
                return
            }
        }
        let inferenceTime_res = (Date().timeIntervalSince(startDate_res) * 1000)/10
        //vgg
        guard let filePath = Bundle.main.path(forResource: "vgg11", ofType: "pt"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        let startDate_vgg = Date()
        for _ in 0 ..< 10 {
            guard let output_vgg = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
                return
            }
        }
        let inferenceTime_vgg = (Date().timeIntervalSince(startDate_vgg) * 1000)/10
        //mobilenet
        guard let filePath = Bundle.main.path(forResource: "mobilenet_v3", ofType: "pt"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        let startDate_mob = Date()
        for _ in 0 ..< 10 {
            guard let output_mob = module.predict_mob(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
                return
            }
        }
        let inferenceTime_mob = (Date().timeIntervalSince(startDate_mob) * 1000)/10
        //convtasnet
        guard let filePath = Bundle.main.path(forResource: "convtasnet", ofType: "pt"),
              let module = TorchModule(fileAtPath: filePath) else { return }
        let startDate_conv = Date()
        for _ in 0 ..< 10 {
            guard let output_conv = module.predict_conv(image: UnsafeMutableRawPointer(&audioBuffer)) else {
                return
            }
        }
        let inferenceTime_conv = (Date().timeIntervalSince(startDate_conv) * 1000)/10
        text += String(format: "Inception_v3: %.2fms", inferenceTime_inc) + "\n\n"
        text += String(format: "Resnet_101: %.2fms", inferenceTime_res) + "\n\n"
        text += String(format: "Mobilenet_v3: %.2fms", inferenceTime_mob) + "\n\n"
        text += String(format: "VGG11: %.2fms", inferenceTime_vgg) + "\n\n"
        text += String(format: "Convtasnet: %.2fms", inferenceTime_conv) + "\n\n"
        text += "\n\n\n\n"
        resultView.text = text
    }
}
