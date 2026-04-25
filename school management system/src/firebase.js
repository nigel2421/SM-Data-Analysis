import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAxlSLqiFAz725HxoPVby9Jli47XV1_F9w",
  authDomain: "school-management-db061.firebaseapp.com",
  projectId: "school-management-db061",
  storageBucket: "school-management-db061.firebasestorage.app",
  messagingSenderId: "1031631076859",
  appId: "1:1031631076859:web:8e14d474970d418aa0df5e",
  measurementId: "G-28VZZFL46T"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Services
export const db = getFirestore(app);
export const auth = getAuth(app);
export const analytics = getAnalytics(app);

export default app;
