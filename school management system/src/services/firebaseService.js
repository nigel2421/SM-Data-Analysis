import { collection, query, where, getDocs, addDoc } from "firebase/firestore";
import { db } from "../firebase";
import * as dummyData from "../utils/dummyData";

// Check if Firebase is actually configured (not using placeholders)
const isFirebaseConfigured = () => {
  try {
    const config = db.app.options;
    return config.apiKey && !config.apiKey.includes("PASTE_YOUR_API_KEY");
  } catch {
    return false;
  }
};

/**
 * Generic Fetcher that handles Multi-tenancy
 */
export const fetchData = async (collectionName, schoolId) => {
  if (!isFirebaseConfigured()) {
    console.warn(`Firebase not configured. Falling back to dummy data for: ${collectionName}`);
    // Map collection names to dummy data exports
    const map = {
      students: dummyData.STUDENTS,
      staff: dummyData.STAFF,
      // Add more mappings as needed
    };
    return map[collectionName] || [];
  }

  try {
    const q = query(collection(db, collectionName), where("schoolId", "==", schoolId));
    const querySnapshot = await getDocs(q);
    return querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
  } catch (error) {
    console.error("Error fetching data: ", error);
    throw error;
  }
};

/**
 * Seed Dummy Data to Firestore (Admin Utility)
 */
export const seedDatabase = async (schoolId) => {
  if (!isFirebaseConfigured()) return;

  const dataToSeed = [
    { name: 'students', data: dummyData.STUDENTS },
    { name: 'staff', data: dummyData.STAFF }
  ];

  for (const item of dataToSeed) {
    for (const record of item.data) {
      await addDoc(collection(db, item.name), {
        ...record,
        schoolId: schoolId
      });
    }
  }
};
