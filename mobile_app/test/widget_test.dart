// Basic Flutter widget test for CropGuard app.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:cropguard/main.dart';

void main() {
  testWidgets('CropGuard app loads', (WidgetTester tester) async {
    await tester.pumpWidget(const CropGuardApp());

    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
