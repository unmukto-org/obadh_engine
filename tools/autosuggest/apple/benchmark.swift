// Native Core ML diagnostic. Compile and benchmark are separate processes so
// conversion/compiler memory is not mistaken for inference memory.
import Foundation
import CoreML
import CryptoKit
import Darwin

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}
struct Fixtures: Decodable {
    let version: Int
    let vocabularySize: Int
    let contexts: [[Int32]]
    let candidates: [[Int32]]
    let checkpointSHA256: String
    let retrievalSHA256: String
}

func memory() -> [String: UInt64] {
    var basic = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &basic) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    var vm = task_vm_info_data_t()
    var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let vmStatus = withUnsafeMutablePointer(to: &vm) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
        }
    }
    var usage = rusage()
    let usageStatus = getrusage(RUSAGE_SELF, &usage)
    var result: [String: UInt64] = [:]
    if status == KERN_SUCCESS { result["resident_bytes"] = basic.resident_size }
    if vmStatus == KERN_SUCCESS {
        result["physical_footprint_bytes"] = vm.phys_footprint
        result["process_peak_resident_bytes"] = vm.resident_size_peak
        if vm.ledger_phys_footprint_peak > 0 {
            result["process_peak_physical_footprint_bytes"] = UInt64(vm.ledger_phys_footprint_peak)
        }
    }
    if usageStatus == 0 { result["rusage_peak_resident_bytes"] = UInt64(usage.ru_maxrss) }
    return result
}
func emit(_ value: [String: Any]) throws {
    let bytes = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .prettyPrinted])
    FileHandle.standardOutput.write(bytes)
    FileHandle.standardOutput.write(Data([10]))
}
func width(_ model: MLModel, _ name: String) throws -> Int {
    guard let constraint = model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint,
          constraint.dataType == .int32, constraint.shape.count == 2,
          constraint.shape[0].intValue == 1, constraint.shape[1].intValue > 0 else {
        throw Failure("Expected fixed batch-one int32 input: \(name)")
    }
    return constraint.shape[1].intValue
}
func buffer(_ values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
    for (index, value) in values.enumerated() { pointer[index * array.strides[1].intValue] = value }
    return array
}
func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    let index = min(sorted.count - 1, max(0, Int(ceil(fraction * Double(sorted.count))) - 1))
    return sorted[index]
}
func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let command = args.first else { throw Failure("Use compile PACKAGE DESTINATION or benchmark COMPILED FIXTURES UNIT ITERATIONS") }
    if command == "compile" {
        guard args.count == 3 else { throw Failure("compile requires package and destination paths") }
        let destination = URL(fileURLWithPath: args[2])
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw Failure("Destination already exists") }
        let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: args[1]))
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: compiled, to: destination)
        try emit(["compiled_model": destination.path])
        return
    }
    guard command == "benchmark", args.count == 5,
          let iterations = Int(args[4]), iterations >= 100, iterations <= 100_000 else {
        throw Failure("benchmark requires compiled model, fixtures, compute unit (all/cpu), and 100...100000 iterations")
    }
    let configuration = MLModelConfiguration()
    switch args[3] {
    case "all": configuration.computeUnits = .all
    case "cpu": configuration.computeUnits = .cpuOnly
    default: throw Failure("Compute unit must be all or cpu")
    }
    let fixtureBytes = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    let fixtures = try JSONDecoder().decode(Fixtures.self, from: fixtureBytes)
    guard fixtures.version == 1, fixtures.vocabularySize > 3,
          !fixtures.contexts.isEmpty, fixtures.contexts.count <= 10_000,
          fixtures.contexts.count == fixtures.candidates.count else { throw Failure("Invalid fixture contract") }
    let baseline = memory()
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let model = try MLModel(contentsOf: URL(fileURLWithPath: args[1]), configuration: configuration)
    let loadUS = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1000
    let afterLoad = memory()
    let contextWidth = try width(model, "contexts")
    let candidateWidth = try width(model, "candidate_ids")
    var providers: [MLDictionaryFeatureProvider] = []
    providers.reserveCapacity(fixtures.contexts.count)
    for index in fixtures.contexts.indices {
        let context = fixtures.contexts[index], candidates = fixtures.candidates[index]
        guard context.count == contextWidth, candidates.count == candidateWidth,
              (context + candidates).allSatisfy({ $0 >= 0 && Int($0) < fixtures.vocabularySize }) else {
            throw Failure("Fixture dimensions or token IDs violate the model contract")
        }
        providers.append(try MLDictionaryFeatureProvider(dictionary: [
            "contexts": MLFeatureValue(multiArray: try buffer(context)),
            "candidate_ids": MLFeatureValue(multiArray: try buffer(candidates))
        ]))
    }
    let afterProviders = memory()
    var durations: [Double] = []
    durations.reserveCapacity(iterations)
    var checksum: Double = 0
    func predict(_ index: Int, measured: Bool) throws {
        try autoreleasepool {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try model.prediction(from: providers[index % providers.count])
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1000
            guard let scores = result.featureValue(for: "scores")?.multiArrayValue,
                  scores.count == candidateWidth else { throw Failure("Invalid score output shape") }
            for j in 0..<scores.count {
                let score = scores[j].doubleValue
                guard score.isFinite else { throw Failure("Non-finite model output") }
                checksum += score
            }
            if measured { durations.append(elapsed) }
        }
    }
    let firstStart = DispatchTime.now().uptimeNanoseconds
    try predict(0, measured: false)
    let firstPredictionUS = Double(DispatchTime.now().uptimeNanoseconds - firstStart) / 1000
    for index in 0..<100 { try predict(index, measured: false) }
    let afterWarmup = memory()
    for index in 0..<iterations { try predict(index, measured: true) }
    let afterRun = memory()
    let sorted = durations.sorted()
    try emit([
        "scope": "native macOS model calls; excludes keyboard scheduling and feature preparation; not an iPhone measurement",
        "compute_units": args[3], "iterations": iterations, "warmup_calls": 101,
        "fixture_cases": providers.count, "context_width": contextWidth, "candidate_width": candidateWidth,
        "compiled_model_path": URL(fileURLWithPath: args[1]).path,
        "fixture_checkpoint_sha256": fixtures.checkpointSHA256, "fixture_retrieval_sha256": fixtures.retrievalSHA256,
        "fixtures_sha256": SHA256.hash(data: fixtureBytes).map { String(format: "%02x", $0) }.joined(),
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "load_us": loadUS, "first_prediction_including_output_validation_us": firstPredictionUS,
        "warm_prediction_us": ["mean": durations.reduce(0, +) / Double(iterations),
                               "p50": percentile(sorted, 0.50), "p95": percentile(sorted, 0.95),
                               "p99": percentile(sorted, 0.99), "max": sorted.last!],
        "memory_bytes": ["baseline": baseline, "after_load": afterLoad, "after_features": afterProviders,
                         "after_warmup": afterWarmup, "after_run": afterRun],
        "output_checksum": checksum
    ])
}
do { try run() }
catch {
    FileHandle.standardError.write(Data("benchmark error: \(error)\n".utf8))
    exit(1)
}
