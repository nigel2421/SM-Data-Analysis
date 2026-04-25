import React, { useRef } from 'react';
import { Printer, Download, Mail } from 'lucide-react';

interface ReceiptProps {
  unitName: String;
  guestName: String;
  amount: String;
  date: String;
  receiptId: String;
}

export const ReceiptGenerator = ({ unitName, guestName, amount, date, receiptId }: ReceiptProps) => {
  const receiptRef = useRef<HTMLDivElement>(null);

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="glass p-8 rounded-2xl max-w-2xl mx-auto border border-white/10 shadow-2xl">
      <div className="flex justify-between items-start mb-8">
        <h2 className="text-2xl font-bold text-white">Generate Receipt</h2>
        <div className="flex gap-2 print:hidden">
          <button 
            onClick={handlePrint}
            className="p-2 glass rounded-lg hover:bg-white/10 text-accent transition-all"
            title="Print Receipt"
          >
            <Printer size={20} />
          </button>
          <button className="p-2 glass rounded-lg hover:bg-white/10 text-slate-400">
            <Download size={20} />
          </button>
          <button className="p-2 glass rounded-lg hover:bg-white/10 text-slate-400">
            <Mail size={20} />
          </button>
        </div>
      </div>

      <div ref={receiptRef} className="bg-white text-slate-900 p-10 rounded-xl shadow-inner font-mono text-sm leading-relaxed border-2 border-slate-100">
        <div className="text-center mb-8 border-b-2 border-slate-100 pb-6">
          <h1 className="text-2xl font-black uppercase tracking-tighter mb-1">MogulPMS</h1>
          <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Premium Property Management</p>
        </div>

        <div className="flex justify-between mb-8">
          <div>
            <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Issued To</p>
            <p className="text-lg font-bold">{guestName}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Receipt ID</p>
            <p className="font-bold">{receiptId}</p>
            <p className="text-xs text-slate-500 mt-1">{date}</p>
          </div>
        </div>

        <div className="mb-10">
          <table className="w-full">
            <thead className="border-b-2 border-slate-100 font-bold text-[10px] uppercase text-slate-400">
              <tr>
                <th className="py-2 text-left">Description</th>
                <th className="py-2 text-right">Amount</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="py-4">Stay at {unitName}</td>
                <td className="py-4 text-right font-bold">{amount}</td>
              </tr>
              <tr className="border-t border-slate-50">
                <td className="py-4 text-slate-500">Service Fee (15%)</td>
                <td className="py-4 text-right text-slate-500">$180.00</td>
              </tr>
            </tbody>
            <tfoot className="border-t-2 border-slate-900">
              <tr>
                <td className="py-4 font-black uppercase">Total Paid</td>
                <td className="py-4 text-right font-black text-lg underline decoration-accent decoration-4 underline-offset-4">{amount}</td>
              </tr>
            </tfoot>
          </table>
        </div>

        <div className="text-center text-[8px] text-slate-300 font-bold uppercase tracking-widest mt-10">
          Thank you for choosing MogulPMS. Have a luxurious stay.
        </div>
      </div>
    </div>
  );
};
