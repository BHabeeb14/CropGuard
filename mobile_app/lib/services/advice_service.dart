/// Localized advice system for crop disease recommendations.
/// Provides offline, actionable guidance based on HCI design.
class AdviceService {
  /// Returns localized advice for a detected disease/health state.
  /// Format: "Plant___Disease" or "Plant___healthy"
  static String getAdvice(String label, double confidence) {
    final isHealthy = label.toLowerCase().contains('healthy');

    if (isHealthy) {
      return 'Your plant appears healthy. Continue regular monitoring and '
          'maintain good farming practices.';
    }

    // Disease-specific advice (expandable for full PlantVillage coverage)
    final advice = _getDiseaseAdvice(label);
    return '$advice\n\nConfidence: ${(confidence * 100).toStringAsFixed(1)}%';
  }

  static String _getDiseaseAdvice(String label) {
    final lower = label.toLowerCase();

    if (lower.contains('scab')) {
      return 'Apple Scab detected. Remove infected leaves. Apply sulfur-based '
          'fungicide. Improve air circulation.';
    }
    if (lower.contains('rot') || lower.contains('blight')) {
      return 'Rot/Blight detected. Remove affected plants. Avoid overhead '
          'watering. Apply copper-based fungicide.';
    }
    if (lower.contains('rust')) {
      return 'Rust detected. Remove infected leaves. Apply fungicide. '
          'Ensure proper spacing between plants.';
    }
    if (lower.contains('mildew')) {
      return 'Powdery mildew detected. Improve air flow. Apply sulfur or '
          'neem oil. Water at soil level.';
    }
    if (lower.contains('spot') || lower.contains('blight')) {
      return 'Leaf spot detected. Remove infected leaves. Apply fungicide. '
          'Avoid wetting foliage.';
    }
    if (lower.contains('virus') || lower.contains('mosaic')) {
      return 'Viral infection suspected. Remove infected plants. Control '
          'aphids and other vectors. Use disease-free seeds.';
    }
    if (lower.contains('bacterial')) {
      return 'Bacterial infection detected. Remove affected parts. '
          'Apply copper spray. Avoid overhead irrigation.';
    }

    return 'Disease detected. Isolate affected plants. Consult local '
        'agricultural extension for region-specific treatment.';
  }

  /// Severity level for color-coded UI.
  static String getSeverity(String label, double confidence) {
    if (label.toLowerCase().contains('healthy')) return 'healthy';
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
  }
}
