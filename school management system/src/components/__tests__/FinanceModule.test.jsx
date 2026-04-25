import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import FinanceModule from '../FinanceModule';

describe('FinanceModule Component', () => {
  it('renders finance header and initial stats', () => {
    render(<FinanceModule />);
    expect(screen.getByText(/Financial Accounts/i)).toBeInTheDocument();
    expect(screen.getByText(/Total Revenue/i)).toBeInTheDocument();
  });

  it('switches between sub-tabs', () => {
    render(<FinanceModule />);
    
    const budgetsTab = screen.getByRole('button', { name: /Budgets/i });
    fireEvent.click(budgetsTab);
    expect(screen.getByText(/Budget Approvals/i)).toBeInTheDocument();
    
    const docsTab = screen.getByRole('button', { name: /Supporting Docs/i });
    fireEvent.click(docsTab);
    expect(screen.getByText(/Document Repository/i)).toBeInTheDocument();
    
    const cashbookTab = screen.getByRole('button', { name: /Fees & Cashbook/i });
    fireEvent.click(cashbookTab);
    expect(screen.getByText(/Daily Cashbook/i)).toBeInTheDocument();
  });

  it('renders transactions table in cashbook view', () => {
    render(<FinanceModule />);
    expect(screen.getByText(/Transaction ID/i)).toBeInTheDocument();
    expect(screen.getByText(/Tuition Fees/i)).toBeInTheDocument();
  });
});
