import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// TFLite inference: hierarchical (crop → disease) when [hierarchy_manifest.json]
/// has `enabled: true`, otherwise single flat classifier.
class InferenceService {
  static const String _manifestPath = 'assets/hierarchy_manifest.json';
  static const String _flatModelPath = 'assets/models/cropguard_model_quantized.tflite';
  static const String _flatLabelsPath = 'assets/labels/class_labels.txt';
  static const int inputSize = 224;

  Interpreter? _flatInterpreter;
  List<String> _flatLabels = [];

  bool _hierarchical = false;
  Interpreter? _cropInterpreter;
  List<String> _cropLabels = [];
  Map<String, dynamic>? _manifest;

  final Map<String, Interpreter> _diseaseInterpreters = {};
  final Map<String, List<String>> _diseaseLabels = {};

  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;
  bool get isWebPlatform => false;
  bool get isHierarchical => _hierarchical;

  Future<void> loadModel() async {
    if (_isLoaded) return;

    try {
      final manifestStr = await rootBundle.loadString(_manifestPath);
      final manifest = jsonDecode(manifestStr) as Map<String, dynamic>;
      _manifest = manifest;

      if (manifest['enabled'] == true) {
        await _loadHierarchical(manifest);
      } else {
        await _loadFlat();
      }
      _isLoaded = true;
      debugPrint(
        'CropGuard: Model loaded (${_hierarchical ? "hierarchical" : "flat"}, '
        '${_hierarchical ? _cropLabels.length : _flatLabels.length} top-level classes)',
      );
    } catch (e) {
      debugPrint('CropGuard: Failed to load model: $e');
      rethrow;
    }
  }

  Future<void> _loadFlat() async {
    _hierarchical = false;
    _flatInterpreter = await Interpreter.fromAsset(_flatModelPath);
    _flatLabels = await _loadLabelsFile(_flatLabelsPath);
    final out = _flatInterpreter!.getOutputTensor(0).shape;
    final n = out.isNotEmpty ? out.last : 38;
    if (_flatLabels.length != n) {
      _flatLabels = List.generate(
        n,
        (i) => i < _flatLabels.length ? _flatLabels[i] : 'Class_$i',
      );
    }
  }

  Future<void> _loadHierarchical(Map<String, dynamic> manifest) async {
    _hierarchical = true;
    final cropModel = manifest['crop_model'] as String? ?? '';
    final cropLabelsPath = manifest['crop_labels'] as String? ?? '';
    if (cropModel.isEmpty || cropLabelsPath.isEmpty) {
      throw StateError('Hierarchical manifest missing crop_model or crop_labels');
    }
    _cropInterpreter = await Interpreter.fromAsset(cropModel);
    _cropLabels = await _loadLabelsFile(cropLabelsPath);
    final out = _cropInterpreter!.getOutputTensor(0).shape;
    final n = out.isNotEmpty ? out.last : _cropLabels.length;
    if (_cropLabels.length != n) {
      _cropLabels = List.generate(
        n,
        (i) => i < _cropLabels.length ? _cropLabels[i] : 'Crop_$i',
      );
    }
  }

  Future<List<String>> _loadLabelsFile(String path) async {
    final data = await rootBundle.loadString(path);
    return data
        .split('\n')
        .where((s) => s.trim().isNotEmpty && !s.startsWith('#'))
        .toList();
  }

  Future<Interpreter> _getDiseaseInterpreter(String assetPath) async {
    if (_diseaseInterpreters.containsKey(assetPath)) {
      return _diseaseInterpreters[assetPath]!;
    }
    final interp = await Interpreter.fromAsset(assetPath);
    _diseaseInterpreters[assetPath] = interp;
    return interp;
  }

  Future<List<String>> _getDiseaseLabels(String labelsPath) async {
    if (_diseaseLabels.containsKey(labelsPath)) {
      return _diseaseLabels[labelsPath]!;
    }
    final lines = await _loadLabelsFile(labelsPath);
    _diseaseLabels[labelsPath] = lines;
    return lines;
  }

  /// Runs flat or hierarchical inference (disease models are loaded lazily).
  Future<List<Map<String, dynamic>>> runInference(
    List<List<List<List<double>>>> input,
  ) async {
    if (!_isLoaded) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }
    if (!_hierarchical) {
      final out = List.generate(1, (_) => List.filled(_flatLabels.length, 0.0));
      _flatInterpreter!.run(input, out);
      return _resultsFromScores(out[0], _flatLabels);
    }

    final manifest = _manifest!;
    final crops = manifest['crops'] as List<dynamic>? ?? [];
    final cropOut = List.generate(1, (_) => List.filled(_cropLabels.length, 0.0));
    _cropInterpreter!.run(input, cropOut);
    final cropScores = cropOut[0];
    final cropIdx = _argmax(cropScores);
    final cropConf = cropScores[cropIdx];
    final cropName = cropIdx < _cropLabels.length ? _cropLabels[cropIdx] : 'Unknown';

    if (cropIdx >= crops.length) {
      return [
        {
          'label': cropName,
          'confidence': cropConf,
          'crop': cropName,
          'disease': '',
        },
      ];
    }

    final entry = crops[cropIdx] as Map<String, dynamic>;
    final disease = entry['disease'] as Map<String, dynamic>? ?? {};
    final mode = disease['mode'] as String? ?? 'single';

    if (mode == 'single') {
      final lbl = disease['label'] as String? ?? 'unknown';
      final full = '$cropName — ${_humanize(lbl)}';
      return [
        {
          'label': full,
          'confidence': cropConf,
          'crop': cropName,
          'disease': lbl,
        },
      ];
    }

    final modelPath = disease['model'] as String? ?? '';
    final labelsPath = disease['labels'] as String? ?? '';
    if (modelPath.isEmpty || labelsPath.isEmpty) {
      return [
        {
          'label': cropName,
          'confidence': cropConf,
          'crop': cropName,
          'disease': '',
        },
      ];
    }

    final interp = await _getDiseaseInterpreter(modelPath);
    var dLabels = await _getDiseaseLabels(labelsPath);
    final outShape = interp.getOutputTensor(0).shape;
    final n = outShape.isNotEmpty ? outShape.last : dLabels.length;
    final out = List.generate(1, (_) => List.filled(n, 0.0));
    interp.run(input, out);
    final dScores = out[0];
    final dIdx = _argmax(dScores);
    final dConf = dIdx < dScores.length ? dScores[dIdx] : 0.0;
    if (dLabels.length != n) {
      dLabels = List.generate(
        n,
        (i) => i < dLabels.length ? dLabels[i] : 'Class_$i',
      );
    }
    final diseaseName = dIdx < dLabels.length ? dLabels[dIdx] : 'Unknown';
    final joint = cropConf * dConf;
    final full = '$cropName — ${_humanize(diseaseName)}';

    return [
      {
        'label': full,
        'confidence': joint,
        'crop': cropName,
        'disease': diseaseName,
        'crop_confidence': cropConf,
        'disease_confidence': dConf,
      },
    ];
  }

  List<Map<String, dynamic>> _resultsFromScores(List<double> scores, List<String> labels) {
    final results = <Map<String, dynamic>>[];
    for (var i = 0; i < scores.length && i < labels.length; i++) {
      results.add({'label': labels[i], 'confidence': scores[i]});
    }
    results.sort((a, b) =>
        (b['confidence'] as double).compareTo(a['confidence'] as double));
    return results;
  }

  int _argmax(List<double> scores) {
    var best = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    return best;
  }

  String _humanize(String s) {
    return s.replaceAll('___', ' ').replaceAll('_', ' ');
  }

  void dispose() {
    _flatInterpreter?.close();
    _flatInterpreter = null;
    _cropInterpreter?.close();
    _cropInterpreter = null;
    for (final i in _diseaseInterpreters.values) {
      i.close();
    }
    _diseaseInterpreters.clear();
    _diseaseLabels.clear();
    _isLoaded = false;
  }
}
