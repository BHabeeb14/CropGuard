/// TFLite inference service - uses native implementation on mobile/desktop,
/// stub on web (dart:ffi not available).
export 'inference_service_stub.dart'
    if (dart.library.ffi) 'inference_service_io.dart';
