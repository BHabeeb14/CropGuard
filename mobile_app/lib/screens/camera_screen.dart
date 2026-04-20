import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import '../services/inference_service.dart';
import '../services/advice_service.dart';
import '../theme/app_theme.dart';

class CameraScreen extends StatefulWidget {
  final InferenceService inferenceService;

  const CameraScreen({super.key, required this.inferenceService});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  /// Minimum confidence to show a result. Below this = likely not a leaf.
  /// 0.35 balances real-world photos vs non-leaf images.
  static const double _confidenceThreshold = 0.35;
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  String? _error;
  Map<String, dynamic>? _lastResult;
  bool _lowConfidence = false; // True when result is below threshold
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        setState(() => _error = 'No camera found');
        return;
      }
      _controller = CameraController(
        _cameras!.first,
        ResolutionPreset.medium,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await _controller!.initialize();
      setState(() {
        _isInitialized = true;
        _error = null;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  Future<void> _captureAndAnalyze() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        _isProcessing) {
      return;
    }

    setState(() => _isProcessing = true);

    try {
      final image = await _controller!.takePicture();
      final bytes = await image.readAsBytes();
      final input = await _preprocessImage(bytes);

      final results = await widget.inferenceService.runInference(input);
      if (results.isNotEmpty) {
        final top = results.first;
        final confidence = top['confidence'] as double;
        setState(() {
          _lastResult = top;
          _lowConfidence = confidence < _confidenceThreshold;
          _isProcessing = false;
        });
      } else {
        setState(() {
          _lastResult = null;
          _lowConfidence = false;
          _isProcessing = false;
        });
      }
    } catch (e) {
      setState(() {
        _lastResult = {'label': 'Error', 'confidence': 0.0};
        _isProcessing = false;
      });
    }
  }

  /// Preprocess camera image to [1, 224, 224, 3] float32.
  /// Model has Rescaling layer expecting [0,255] → must use true.
  static const bool _inputRange0_255 = true;

  Future<List<List<List<List<double>>>>> _preprocessImage(
      Uint8List bytes) async {
    var decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception('Failed to decode image');

    // Apply EXIF orientation (e.g. phone rotated)
    decoded = img.bakeOrientation(decoded);

    // Resize with linear interpolation to match TensorFlow's bilinear
    final resized = img.copyResize(
      decoded,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );

    // Shape [1, 224, 224, 3] - RGB, range must match training pipeline
    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resized.getPixel(x, y);
            final r = pixel.r.toDouble();
            final g = pixel.g.toDouble();
            final b = pixel.b.toDouble();
            return _inputRange0_255
                ? [r, g, b] // [0, 255] - matches training
                : [
                    r / 255,
                    g / 255,
                    b / 255
                  ]; // [0, 1] - fallback if model expects this
          },
        ),
      ),
    );
    return input;
  }

  Color _getSeverityColor(String severity) {
    switch (severity) {
      case 'healthy':
        return DiseaseSeverity.healthy;
      case 'high':
        return DiseaseSeverity.high;
      case 'medium':
        return DiseaseSeverity.medium;
      default:
        return DiseaseSeverity.low;
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Scan Plant')),
        body: Center(child: Text(_error!)),
      );
    }

    if (!_isInitialized || _controller == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Scan Plant')),
      body: Column(
        children: [
          Expanded(
            child: CameraPreview(_controller!),
          ),
          if (_lastResult != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              color: _lowConfidence
                  ? Colors.amber.withValues(alpha: 0.3)
                  : _getSeverityColor(
                      AdviceService.getSeverity(
                        _lastResult!['label'] as String,
                        _lastResult!['confidence'] as double,
                      ),
                    ).withValues(alpha: 0.3),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _lowConfidence
                        ? 'Unable to identify'
                        : (_lastResult!['label'] as String)
                            .replaceAll('___', ' ')
                            .replaceAll('_', ' '),
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _lowConfidence
                        ? 'Please capture a clear image of a plant leaf. The model is trained on leaf images only.'
                        : AdviceService.getAdvice(
                            _lastResult!['label'] as String,
                            _lastResult!['confidence'] as double,
                          ),
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
          Padding(
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _isProcessing ? null : _captureAndAnalyze,
                icon: _isProcessing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.camera),
                label:
                    Text(_isProcessing ? 'Analyzing...' : 'Capture & Analyze'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
