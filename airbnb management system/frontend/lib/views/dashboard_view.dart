import 'package:flutter/material.dart';

class DashboardView extends StatelessWidget {
  const DashboardView({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('MogulPMS Command Center'),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          const Center(
            child: Text(
              'Global Map View Loading...',
              style: TextStyle(color: Colors.grey),
            ),
          ),
          // Mock Unit Pins
          Positioned(
            bottom: 200,
            left: 50,
            child: _UnitPin(status: Colors.green, label: 'Unit 4A'),
          ),
          Positioned(
            bottom: 150,
            right: 80,
            child: _UnitPin(status: Colors.red, label: 'Unit 2B'),
          ),
          Positioned(
            top: 200,
            left: 100,
            child: _UnitPin(status: Colors.yellow, label: 'Studio C'),
          ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        backgroundColor: const Color(0xFF1A1A2E),
        selectedItemColor: Colors.tealAccent,
        unselectedItemColor: Colors.white54,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.map), label: 'Map'),
          BottomNavigationBarItem(icon: Icon(Icons.list), label: 'Units'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Account'),
        ],
      ),
    );
  }
}

class _UnitPin extends StatelessWidget {
  final Color status;
  final String label;

  const _UnitPin({required this.status, required this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(Icons.location_on, color: status, size: 40),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(label, style: const TextStyle(fontSize: 10)),
        ),
      ],
    );
  }
}
