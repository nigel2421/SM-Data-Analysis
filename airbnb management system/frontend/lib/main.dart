import 'package:flutter/material.dart';
import 'views/dashboard_view.dart';

void main() {
  runApp(const MogulPMSApp());
}

class MogulPMSApp extends StatelessWidget {
  const MogulPMSApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MogulPMS',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.teal,
        scaffoldBackgroundColor: const Color(0xFF0F0F1A),
      ),
      home: const DashboardView(),
    );
  }
}
