import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

/// Parsed crop + condition from a raw model label.
///
/// Examples of raw labels and what we extract:
///   Apple___Apple_scab               -> crop "Apple",   condition "Apple scab"
///   Tomato___Tomato_Yellow_Leaf_...  -> crop "Tomato",  condition "Tomato Yellow Leaf Curl Virus"
///   CPDD_Cassava_mosaic              -> crop "Cassava", condition "mosaic"
///   Corn_(maize)___Common_rust_      -> crop "Corn (maize)", condition "Common rust"
///   Pepper,_bell___Bacterial_spot    -> crop "Pepper, bell", condition "Bacterial spot"
class ParsedLabel {
  final String raw;
  final String crop;
  final String condition;
  final bool isHealthy;

  const ParsedLabel({
    required this.raw,
    required this.crop,
    required this.condition,
    required this.isHealthy,
  });
}

/// Structured, localisable advice for a single class label.
///
/// Decoupled from the code so that extending coverage or translating
/// the text does not require a new build of business logic.
class Advice {
  final String summary;
  final List<String> immediate;
  final List<String> treatmentOrganic;
  final List<String> treatmentChemical;
  final List<String> prevention;

  /// Intrinsic severity of the condition itself
  /// ('healthy' | 'low' | 'medium' | 'high').
  /// Independent of how confident the model is.
  final String severity;

  /// When the farmer should re-scan / re-assess.
  final int followUpDays;

  const Advice({
    required this.summary,
    required this.immediate,
    required this.treatmentOrganic,
    required this.treatmentChemical,
    required this.prevention,
    required this.severity,
    required this.followUpDays,
  });

  factory Advice.fromJson(Map<String, dynamic> json) {
    final treatment = (json['treatment'] as Map?) ?? const {};
    return Advice(
      summary: (json['summary'] as String?) ?? '',
      immediate: _stringList(json['immediate']),
      treatmentOrganic: _stringList(treatment['organic']),
      treatmentChemical: _stringList(treatment['chemical']),
      prevention: _stringList(json['prevention']),
      severity: (json['severity'] as String?) ?? 'low',
      followUpDays: (json['follow_up_days'] as num?)?.toInt() ?? 7,
    );
  }

  static List<String> _stringList(dynamic v) {
    if (v is List) return v.map((e) => e.toString()).toList();
    return const [];
  }
}

/// Loads and serves structured advice for detected labels.
///
/// Call [AdviceService.instance.load()] once at app startup (it is idempotent).
/// Subsequent calls are synchronous and return the cached [Advice] object.
///
/// If the JSON asset is missing or an unknown label comes in, the service
/// falls back to a conservative generic entry rather than throwing.
class AdviceService {
  AdviceService._();
  static final AdviceService instance = AdviceService._();

  static const String _assetPath = 'assets/advice/advice.json';

  Map<String, Advice> _byLabel = const {};
  Advice? _fallbackHealthy;
  Advice? _fallbackUnknown;
  bool _loaded = false;

  Future<void> load() async {
    if (_loaded) return;
    try {
      final raw = await rootBundle.loadString(_assetPath);
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      final map = <String, Advice>{};
      decoded.forEach((key, value) {
        if (key.startsWith('_')) return;
        if (value is Map<String, dynamic>) {
          map[key] = Advice.fromJson(value);
        }
      });
      _byLabel = map;
      _fallbackHealthy = decoded['_fallback_healthy'] is Map<String, dynamic>
          ? Advice.fromJson(
              decoded['_fallback_healthy'] as Map<String, dynamic>)
          : null;
      _fallbackUnknown = decoded['_fallback_unknown'] is Map<String, dynamic>
          ? Advice.fromJson(
              decoded['_fallback_unknown'] as Map<String, dynamic>)
          : null;
    } catch (_) {
      _byLabel = const {};
    }
    _loaded = true;
  }

  /// Returns a structured [Advice] for the given raw model label.
  Advice adviceFor(String label) {
    final exact = _byLabel[label];
    if (exact != null) return exact;

    final parsed = parseLabel(label);
    if (parsed.isHealthy) {
      return _fallbackHealthy ??
          const Advice(
            summary: 'Plant appears healthy.',
            immediate: ['Continue regular monitoring'],
            treatmentOrganic: [],
            treatmentChemical: [],
            prevention: ['Scout weekly'],
            severity: 'healthy',
            followUpDays: 7,
          );
    }
    return _fallbackUnknown ??
        const Advice(
          summary: 'Condition detected but no specific advice available.',
          immediate: ['Isolate affected plants', 'Retake a clearer photo'],
          treatmentOrganic: [],
          treatmentChemical: [],
          prevention: ['Contact your agricultural extension officer'],
          severity: 'low',
          followUpDays: 3,
        );
  }

  /// Intrinsic disease severity band: 'healthy' | 'low' | 'medium' | 'high'.
  ///
  /// This is independent of model confidence. A low-severity leaf miner
  /// identified with 99% confidence is still low-severity; a high-severity
  /// late blight identified with 60% confidence is still high-severity.
  String diseaseSeverity(String label) => adviceFor(label).severity;

  /// Band describing *model* certainty.
  /// Use alongside (not as a substitute for) [diseaseSeverity].
  static String confidenceBand(double confidence) {
    if (confidence >= 0.85) return 'high';
    if (confidence >= 0.65) return 'medium';
    return 'low';
  }

  /// Split a raw model label like `Apple___Apple_scab` or `CPDD_Cassava_mosaic`
  /// into a human-readable crop and condition.
  static ParsedLabel parseLabel(String raw) {
    String crop;
    String condition;

    if (raw.contains('___')) {
      final parts = raw.split('___');
      crop = parts.first;
      condition = parts.sublist(1).join(' ');
    } else if (raw.startsWith('CPDD_')) {
      final rest = raw.substring('CPDD_'.length);
      final idx = rest.indexOf('_');
      if (idx > 0) {
        crop = rest.substring(0, idx);
        condition = rest.substring(idx + 1);
      } else {
        crop = rest;
        condition = '';
      }
    } else {
      crop = raw;
      condition = '';
    }

    String prettify(String s) =>
        s.replaceAll('_', ' ').replaceAll(RegExp(r'\s+'), ' ').trim();

    final prettyCrop = prettify(crop);
    final prettyCondition = prettify(condition);
    final isHealthy = raw.toLowerCase().contains('healthy');

    return ParsedLabel(
      raw: raw,
      crop: prettyCrop,
      condition: isHealthy ? 'healthy' : prettyCondition,
      isHealthy: isHealthy,
    );
  }

  // ---------------------------------------------------------------------------
  // Backward-compatible static API (kept so existing callers keep working).
  // Prefer the structured [adviceFor] / [diseaseSeverity] API for new code.
  // ---------------------------------------------------------------------------

  /// Legacy: free-form advice string. Uses the loaded JSON if available,
  /// otherwise returns a generic fallback.
  static String getAdvice(String label, double confidence) {
    final a = instance.adviceFor(label);
    final conf = '${(confidence * 100).toStringAsFixed(1)}%';
    if (a.severity == 'healthy') {
      return '${a.summary}\n\nConfidence: $conf';
    }
    final lines = <String>[a.summary];
    if (a.immediate.isNotEmpty) {
      lines.add('');
      lines.add('Do now:');
      lines.addAll(a.immediate.map((e) => '• $e'));
    }
    lines.add('');
    lines.add('Confidence: $conf');
    return lines.join('\n');
  }

  /// Legacy: returns the intrinsic disease severity, not a confidence band.
  static String getSeverity(String label, double confidence) {
    return instance.diseaseSeverity(label);
  }
}
