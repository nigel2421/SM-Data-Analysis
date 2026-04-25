import 'package:flutter/material.dart';

class UnitDetailView extends StatelessWidget {
  final String unitName;
  final String status;

  const UnitDetailView({Key? key, required this.unitName, required this.status})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(unitName)),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _InfoCard(
              title: 'Status',
              value: status,
              color: status == 'Occupied' ? Colors.green : Colors.red,
            ),
            const SizedBox(height: 20),
            const Text('Turnover Countdown', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const Text('2h 15m remaining', style: TextStyle(fontSize: 24, color: Colors.yellowAccent)),
            const Divider(height: 40),
            _ToggleSetting(title: 'Dynamic Pricing', value: true),
            const Spacer(),
            ElevatedButton.icon(
              onPressed: () {},
              icon: const Icon(Icons.report_problem),
              label: const Text('Log Maintenance'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
                minimumSize: const Size(double.infinity, 50),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoCard extends StatelessWidget {
  final String title;
  final String value;
  final Color color;

  const _InfoCard({required this.title, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A2E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(color: Colors.white54)),
          const SizedBox(height: 8),
          Text(value, style: TextStyle(fontSize: 22, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

class _ToggleSetting extends StatelessWidget {
  final String title;
  final bool value;

  const _ToggleSetting({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title, style: const TextStyle(fontSize: 18)),
        Switch(value: value, onChanged: (v) {}, activeColor: Colors.tealAccent),
      ],
    );
  }
}
