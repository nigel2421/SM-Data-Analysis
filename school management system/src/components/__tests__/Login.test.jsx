import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Login from '../Login';

// Mock Firebase
vi.mock('../../firebase', () => ({
  auth: {}
}));

vi.mock('firebase/auth', () => ({
  signInWithEmailAndPassword: vi.fn(),
  getAuth: vi.fn()
}));

describe('Login Component', () => {
  it('renders landing page with login button', () => {
    render(<Login />);
    expect(screen.getByText(/Transforming Kenyan Schools/i)).toBeInTheDocument();
    expect(screen.getByText(/Login to Dashboard/i)).toBeInTheDocument();
  });

  it('opens login modal when login button is clicked', () => {
    render(<Login />);
    const loginButton = screen.getByText(/Login to Dashboard/i);
    fireEvent.click(loginButton);
    expect(screen.getByText(/Welcome Back/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/admin@mzalendo.com/i)).toBeInTheDocument();
  });

  it('updates email and password fields', () => {
    render(<Login />);
    fireEvent.click(screen.getByText(/Login to Dashboard/i));
    
    const emailInput = screen.getByPlaceholderText(/admin@mzalendo.com/i);
    const passwordInput = screen.getByPlaceholderText(/••••••••/i);
    
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    
    expect(emailInput.value).toBe('test@example.com');
    expect(passwordInput.value).toBe('password123');
  });
});
