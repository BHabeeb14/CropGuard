import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// CropGuard HCI design - color-coded indicators and localized advice.
/// Severity colors for disease detection results.
class DiseaseSeverity {
  static const Color healthy = Color(0xFF2E7D32);   // Green
  static const Color low = Color(0xFFF9A825);        // Amber
  static const Color medium = Color(0xFFEF6C00);     // Orange
  static const Color high = Color(0xFFC62828);       // Red
}

class AppTheme {
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xFF1B5E20),
        brightness: Brightness.light,
        primary: const Color(0xFF2E7D32),
        secondary: const Color(0xFF558B2F),
      ),
      textTheme: GoogleFonts.poppinsTextTheme(),
      appBarTheme: AppBarTheme(
        elevation: 0,
        centerTitle: true,
        backgroundColor: const Color(0xFF1B5E20),
        foregroundColor: Colors.white,
        titleTextStyle: GoogleFonts.poppins(
          fontSize: 20,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 2,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        color: Colors.white,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF2E7D32),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
    );
  }
}
