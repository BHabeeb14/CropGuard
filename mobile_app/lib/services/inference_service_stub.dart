/// Stub implementation for web - TFLite (dart:ffi) is not available on web.
/// ML runs on Android; web shows UI and prompts for mobile.
class InferenceService {
  bool get isLoaded => false;
  bool get isWebPlatform => true;

  Future<void> loadModel() async {
    // No-op on web; ML requires native platform
  }

  Future<List<Map<String, dynamic>>> runInference(
    List<List<List<List<double>>>> input,
  ) async {
    throw UnsupportedError('Use the Android app for disease detection.');
  }

  void dispose() {}
}
