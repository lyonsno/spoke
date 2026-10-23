// Provisional typed capture producer for the existing Python House renderer.
import AppKit
import ScreenCaptureKit
import CoreMedia
import QuartzCore

public typealias StateCallback = @convention(c) (UnsafePointer<CChar>?) -> Void
public typealias FrameCallback = @convention(c) (UnsafeMutableRawPointer, Int32, Int32, Double) -> Void

private struct Request: Decodable {
    let x, y, widthPoints, heightPoints: Double
    let width, height, fps: Int
    let excludedWindowID: UInt32
    let sourceWindowID: UInt32
}

private final class Output: NSObject, SCStreamOutput {
    let frame: FrameCallback
    init(_ frame: @escaping FrameCallback) { self.frame = frame }
    func stream(_ stream: SCStream, didOutputSampleBuffer sample: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .screen,
              let info = (CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: false)
                           as? [[SCStreamFrameInfo: Any]])?.first,
              (info[.status] as? NSNumber)?.intValue == SCFrameStatus.complete.rawValue,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { return }
        var source = 0.0
        if let ticks = info[.displayTime] as? NSNumber {
            var timebase = mach_timebase_info_data_t()
            mach_timebase_info(&timebase)
            source = ticks.doubleValue * Double(timebase.numer) / Double(timebase.denom) / 1e9
        }
        frame(Unmanaged.passUnretained(pixelBuffer).toOpaque(),
              Int32(CVPixelBufferGetWidth(pixelBuffer)), Int32(CVPixelBufferGetHeight(pixelBuffer)), source)
    }
}

private var active: (SCStream, Output)?
private var cachedContent: SCShareableContent?
private let stateLock = NSLock()

private func trace(_ phase: String) {
    FileHandle.standardError.write(Data("typed-capture: \(phase)\n".utf8))
}

private func result(_ callback: StateCallback, _ error: Error?) {
    if let error = error { String(describing: error).withCString { callback($0) } }
    else { callback(nil) }
}

@_cdecl("spoke_capture_start")
public func start(_ json: UnsafePointer<CChar>, _ frame: @escaping FrameCallback,
                  _ state: @escaping StateCallback) {
    let data = Data(String(cString: json).utf8)
    trace("start called")
    Task.detached {
        do {
            guard stateLock.withLock({ active == nil }) else { throw NSError(domain: "Capture already active", code: 1) }
            let request = try JSONDecoder().decode(Request.self, from: data)
            trace("enumeration requested")
            let content: SCShareableContent
            if let cached = stateLock.withLock({ cachedContent }) { content = cached }
            else {
                content = try await SCShareableContent.current
                stateLock.withLock { cachedContent = content }
            }
            trace("enumeration returned")
            guard let display = content.displays.first(where: { $0.displayID == CGMainDisplayID() }) else {
                throw NSError(domain: "Main display unavailable", code: 2)
            }
            guard let application = content.applications.first(where: {
                $0.processID == ProcessInfo.processInfo.processIdentifier
            }), let source = content.windows.first(where: { $0.windowID == request.sourceWindowID }) else {
                throw NSError(domain: "Fixture source identity unresolved", code: 3)
            }
            let config = SCStreamConfiguration()
            config.sourceRect = CGRect(x: request.x, y: request.y,
                                      width: request.widthPoints, height: request.heightPoints)
            config.width = request.width; config.height = request.height
            config.pixelFormat = kCVPixelFormatType_32BGRA
            config.queueDepth = 8; config.showsCursor = false
            config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(request.fps))
            let output = Output(frame)
            let stream = SCStream(filter: SCContentFilter(display: display,
                                      excludingApplications: [application], exceptingWindows: [source]),
                                  configuration: config, delegate: nil)
            try stream.addStreamOutput(output, type: .screen,
                                       sampleHandlerQueue: DispatchQueue(label: "spoke.local-liquid.typed"))
            stateLock.withLock { active = (stream, output) }
            trace("stream start requested")
            try await stream.startCapture()
            trace("stream start acknowledged")
            result(state, nil)
        } catch { result(state, error) }
    }
}

@_cdecl("spoke_capture_stop")
public func stop(_ state: @escaping StateCallback) {
    Task.detached {
        do {
            let previous = stateLock.withLock { let value = active; active = nil; return value }
            if let (stream, _) = previous { try await stream.stopCapture() }
            trace("stream stop acknowledged")
            result(state, nil)
        } catch { result(state, error) }
    }
}
