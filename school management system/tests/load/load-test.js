import http from 'k6/http';
import { sleep, check } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 50 },  // Ramp up to 50 users
    { duration: '3m', target: 50 },  // Stay at 50 users
    { duration: '1m', target: 200 }, // "Balloon" to 200 users
    { duration: '3m', target: 200 }, // Stay at 200 users
    { duration: '1m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must be below 500ms
    http_req_failed: ['rate<0.01'],    // Less than 1% failure rate
  },
};

export default function () {
  const BASE_URL = 'http://school-mgmt.mzalendo.com'; // Placeholder URL

  // 1. Visit Landing Page
  const res1 = http.get(BASE_URL);
  check(res1, {
    'landing page status is 200': (r) => r.status === 200,
  });
  sleep(1);

  // 2. Simulate Login (Check static assets)
  const res2 = http.get(`${BASE_URL}/assets/hero.png`);
  check(res2, {
    'hero image status is 200': (r) => r.status === 200,
  });
  sleep(Math.random() * 3 + 1);

  // 3. Navigate to Modules (Simulated by checking asset/manifest)
  const res3 = http.get(`${BASE_URL}/main.js`);
  check(res3, {
    'main bundle status is 200': (r) => r.status === 200,
  });
  sleep(2);
}
