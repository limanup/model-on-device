// Load a converted .mlpackage and run one inference.
//
//   swiftc -O swift/SmokeTest.swift -o smoke
//   ./smoke models/Phoneme-FP16.mlpackage
//
// This exists so the artifact is demonstrably consumable, not just
// convertible. It feeds one second of silence and prints the output shape.

import CoreML
import Foundation

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write("usage: smoke <path/to/Model.mlpackage>\n".data(using: .utf8)!)
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let configuration = MLModelConfiguration()
configuration.computeUnits = .all

let model: MLModel
do {
    model = try MLModel(contentsOf: url, configuration: configuration)
} catch {
    FileHandle.standardError.write("failed to load model: \(error)\n".data(using: .utf8)!)
    exit(1)
}

let length = 16_000  // one second at 16 kHz
let input: MLMultiArray
do {
    input = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float32)
    for index in 0..<length {
        input[index] = 0
    }
} catch {
    FileHandle.standardError.write("failed to build input: \(error)\n".data(using: .utf8)!)
    exit(1)
}

let provider: MLFeatureProvider
do {
    provider = try MLDictionaryFeatureProvider(
        dictionary: ["input_values": MLFeatureValue(multiArray: input)]
    )
} catch {
    FileHandle.standardError.write("failed to build feature provider: \(error)\n".data(using: .utf8)!)
    exit(1)
}

let started = Date()
let output: MLFeatureProvider
do {
    output = try model.prediction(from: provider)
} catch {
    FileHandle.standardError.write("prediction failed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
let elapsed = Date().timeIntervalSince(started)

for name in output.featureNames.sorted() {
    guard let value = output.featureValue(for: name)?.multiArrayValue else { continue }
    let shape = value.shape.map(String.init).joined(separator: " × ")
    print("\(name): \(shape)")
}
print(String(format: "first inference: %.1f ms", elapsed * 1000))
