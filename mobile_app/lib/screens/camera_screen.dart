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

  Widget _buildResultPanel(BuildContext context) {
    final result = _lastResult!;
    final rawLabel = result['label'] as String;
    final confidence = result['confidence'] as double;
    final confidencePct = (confidence * 100).toStringAsFixed(1);

    if (_lowConfidence) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        color: Colors.amber.withValues(alpha: 0.3),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Unable to identify',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 4),
            const Text(
              'Please capture a clear image of a plant leaf. The model is '
              'trained on leaf images only.',
            ),
            const SizedBox(height: 8),
            Text(
              'Model confidence: $confidencePct%',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      );
    }

    final parsed = AdviceService.parseLabel(rawLabel);
    final advice = AdviceService.instance.adviceFor(rawLabel);
    final severityColor = _getSeverityColor(advice.severity);
    final confidenceBand = AdviceService.confidenceBand(confidence);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      color: severityColor.withValues(alpha: 0.15),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  parsed.isHealthy
                      ? '${parsed.crop} — healthy'
                      : '${parsed.crop} — ${parsed.condition}',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ),
              _Chip(
                label: 'Severity: ${advice.severity}',
                color: severityColor,
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(advice.summary, style: Theme.of(context).textTheme.bodyMedium),
          if (advice.immediate.isNotEmpty) ...[
            const SizedBox(height: 12),
            _Section(title: 'Do now', items: advice.immediate),
          ],
          if (advice.treatmentOrganic.isNotEmpty) ...[
            const SizedBox(height: 8),
            _Section(title: 'Organic treatment', items: advice.treatmentOrganic),
          ],
          if (advice.treatmentChemical.isNotEmpty) ...[
            const SizedBox(height: 8),
            _Section(
              title: 'Chemical treatment (follow label)',
              items: advice.treatmentChemical,
            ),
          ],
          if (advice.prevention.isNotEmpty) ...[
            const SizedBox(height: 8),
            _Section(title: 'Prevent recurrence', items: advice.prevention),
          ],
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 4,
            children: [
              _Chip(
                label: 'Model confidence: $confidencePct% ($confidenceBand)',
                color: Colors.blueGrey,
              ),
              _Chip(
                label: 'Re-scan in ${advice.followUpDays} days',
                color: Colors.blueGrey,
              ),
            ],
          ),
        ],
      ),
    );
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
          if (_lastResult != null)
            Expanded(
              child: SingleChildScrollView(
                child: _buildResultPanel(context),
              ),
            ),
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

class _Section extends StatelessWidget {
  final String title;
  final List<String> items;

  const _Section({required this.title, required this.items});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: Theme.of(context)
              .textTheme
              .labelLarge
              ?.copyWith(fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 2),
        ...items.map(
          (e) => Padding(
            padding: const EdgeInsets.symmetric(vertical: 1),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('• '),
                Expanded(
                  child: Text(
                    e,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color color;

  const _Chip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.6)),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }
}
