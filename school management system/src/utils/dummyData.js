export const SCHOOL_DATA = {
  id: 'school_001',
  name: 'Mzalendo Senior Academy',
  location: 'Nairobi, Kenya',
  curriculum: 'CBE',
};

export const GRADES = [
  { id: 'pp1', name: 'Pre-Primary 1', level: 'Early Years' },
  { id: 'pp2', name: 'Pre-Primary 2', level: 'Early Years' },
  { id: 'g1', name: 'Grade 1', level: 'Lower Primary' },
  { id: 'g2', name: 'Grade 2', level: 'Lower Primary' },
  { id: 'g3', name: 'Grade 3', level: 'Lower Primary' },
  { id: 'g4', name: 'Grade 4', level: 'Upper Primary' },
  { id: 'g5', name: 'Grade 5', level: 'Upper Primary' },
  { id: 'g6', name: 'Grade 6', level: 'Upper Primary' },
  { id: 'g7', name: 'Grade 7', level: 'Junior School' },
  { id: 'g8', name: 'Grade 8', level: 'Junior School' },
  { id: 'g9', name: 'Grade 9', level: 'Junior School' },
  { id: 'g10', name: 'Grade 10', level: 'Senior School' },
  { id: 'g11', name: 'Grade 11', level: 'Senior School' },
  { id: 'g12', name: 'Grade 12', level: 'Senior School' },
];

export const SUBJECTS_MAP = {
  'Early Years': ['Literacy', 'Mathematics', 'Creative Arts', 'Environmental Activities', 'Religious Education'],
  'Lower Primary': ['English', 'Kiswahili', 'Mathematics', 'Environmental Activities', 'Hygiene & Nutrition', 'Religious Education', 'P.E.'],
  'Upper Primary': ['English', 'Kiswahili', 'Mathematics', 'Science & Technology', 'Social Studies', 'Agriculture', 'Home Science', 'Creative Arts'],
  'Junior School': ['English', 'Kiswahili', 'Mathematics', 'Integrated Science', 'Health Education', 'Pre-Technical Education', 'Social Studies', 'Life Skills'],
  'Senior School': {
    'STEM': ['Pure Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science'],
    'Social Sciences': ['History', 'Geography', 'Business Studies', 'Legal Studies'],
    'Arts & Sports': ['Music', 'Visual Arts', 'Sports Science', 'Media Studies']
  }
};

export const STUDENTS = [
  { id: 'st001', name: 'Juma Kiptoo', age: 12, grade: 'g7', stream: 'West', parent: 'Mary Kiptoo', balance: 4500, status: 'Active' },
  { id: 'st002', name: 'Amara Wanjiku', age: 10, grade: 'g5', stream: 'North', parent: 'Peter Wanjiku', balance: 0, status: 'Active' },
  { id: 'st003', name: 'Liam Mutua', age: 6, grade: 'g1', stream: 'East', parent: 'Sara Mutua', balance: 12000, status: 'Inactive' },
  { id: 'st004', name: 'Zahra Omar', age: 15, grade: 'g9', stream: 'West', parent: 'Omar Hassan', balance: 2500, status: 'Active' },
];

export const STAFF = [
  { id: 'sf001', name: 'Dr. John Kamau', role: 'Principal', salary: 120000, joined: '2020-01-15' },
  { id: 'sf002', name: 'Jane Doe', role: 'Head of Academics', salary: 95000, joined: '2021-05-10' },
  { id: 'sf003', name: 'Samuel Mwangi', role: 'Bursar', salary: 75000, joined: '2019-11-20' },
];
