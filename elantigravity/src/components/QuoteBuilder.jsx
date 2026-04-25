import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { ChevronRight, ChevronLeft, Upload, CheckCircle2, Send, Mail, Phone, MapPin } from "lucide-react";
import { cn } from "../lib/utils";

// Form Schema
const quoteSchema = z.object({
  service: z.string().min(1, "Please select a service"),
  quantity: z.string().min(1, "Quantity is required"),
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Invalid email address"),
  phone: z.string().min(10, "Invalid phone number"),
  details: z.string().optional(),
});

const services = ["Outdoor Advertising", "Corporate Identity", "Apparel & Merchandise", "Other"];

export default function QuoteBuilder() {
  const [step, setStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const { register, handleSubmit, formState: { errors }, trigger, watch } = useForm({
    resolver: zodResolver(quoteSchema),
    defaultValues: {
      service: "",
      quantity: "",
      name: "",
      email: "",
      phone: "",
      details: "",
    }
  });

  const nextStep = async () => {
    let fieldsToValidate = [];
    if (step === 1) fieldsToValidate = ["service", "quantity"];
    if (step === 2) fieldsToValidate = ["details"]; // Optional but just in case
    
    const isValid = await trigger(fieldsToValidate);
    if (isValid) setStep(s => s + 1);
  };

  const prevStep = () => setStep(s => s - 1);

  const onSubmit = async (data) => {
    setIsSubmitting(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log("Form Data:", data);
    setIsSubmitting(false);
    setIsSuccess(true);
  };

  if (isSuccess) {
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-card p-12 text-center"
      >
        <div className="w-20 h-20 bg-brand-gold/20 rounded-full flex items-center justify-center mx-auto mb-8 border-2 border-brand-gold">
          <CheckCircle2 size={40} className="text-brand-gold" />
        </div>
        <h2 className="text-4xl mb-6 font-display text-brand-black dark:text-brand-white-off transition-colors">Quote Requested!</h2>
        <p className="text-brand-black/60 dark:text-brand-white-off/60 text-lg mb-8 leading-relaxed transition-colors">
          Your request has been received. Our luxury branding experts will review your details and contact you within 24 hours.
        </p>
        <button 
          onClick={() => { setIsSuccess(false); setStep(1); }}
          className="btn-gold"
        >
          Send Another Request
        </button>
      </motion.div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      {/* Progress Bar */}
      <div className="h-1 bg-brand-black/5 dark:bg-brand-white/5 w-full">
        <motion.div 
          initial={{ width: "33.33%" }}
          animate={{ width: `${(step / 3) * 100}%` }}
          className="h-full bg-gold-gradient" 
        />
      </div>

      <div className="p-8 md:p-12">
        <form onSubmit={handleSubmit(onSubmit)}>
          <AnimatePresence mode="wait">
            {step === 1 && (
              <motion.div
                key="step1"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-8"
              >
                <div>
                  <h3 className="text-2xl mb-6 font-display text-brand-black dark:text-brand-white-off transition-colors">Project Scope</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Select Service</label>
                      <select 
                        {...register("service")}
                        className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off focus:outline-none focus:border-brand-gold transition-colors appearance-none"
                      >
                        <option value="" className="bg-brand-white dark:bg-brand-black text-brand-black dark:text-brand-white-off">Choose a service...</option>
                        {services.map(s => <option key={s} value={s} className="bg-brand-white dark:bg-brand-black text-brand-black dark:text-brand-white-off">{s}</option>)}
                      </select>
                      {errors.service && <span className="text-red-500 text-xs">{errors.service.message}</span>}
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Estimated Quantity</label>
                      <input 
                        type="text" 
                        placeholder="e.g. 50 units, 1 billboard..." 
                        {...register("quantity")}
                        className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off placeholder:text-brand-black/30 dark:placeholder:text-brand-white-off/30 focus:outline-none focus:border-brand-gold transition-colors"
                      />
                      {errors.quantity && <span className="text-red-500 text-xs">{errors.quantity.message}</span>}
                    </div>
                  </div>
                </div>
                <button type="button" onClick={nextStep} className="btn-gold w-full flex items-center justify-center">
                  Next Step <ChevronRight size={20} className="ml-2" />
                </button>
              </motion.div>
            )}

            {step === 2 && (
              <motion.div
                key="step2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-8"
              >
                <div>
                  <h3 className="text-2xl mb-6 font-display text-brand-black dark:text-brand-white-off transition-colors">Creative Assets</h3>
                  <div className="border-2 border-dashed border-brand-black/10 dark:border-brand-white/10 rounded-2xl p-12 text-center hover:border-brand-gold/40 transition-colors cursor-pointer group">
                    <Upload size={48} className="mx-auto text-brand-black/20 dark:text-brand-white-off/20 mb-4 group-hover:text-brand-gold transition-colors" />
                    <p className="text-brand-black/60 dark:text-brand-white-off/60 font-medium mb-2 transition-colors">Upload your logo or brand guidelines</p>
                    <p className="text-xs text-brand-black/30 dark:text-brand-white-off/30 uppercase tracking-widest transition-colors">Supports PDF, SVG, AI, PNG (Max 10MB)</p>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Project Details (Optional)</label>
                  <textarea 
                    rows={4} 
                    placeholder="Tell us about your vision..." 
                    {...register("details")}
                    className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off placeholder:text-brand-black/30 dark:placeholder:text-brand-white-off/30 focus:outline-none focus:border-brand-gold transition-colors"
                  />
                </div>
                <div className="flex gap-4">
                  <button type="button" onClick={prevStep} className="btn-outline flex-1">Back</button>
                  <button type="button" onClick={nextStep} className="btn-gold flex-1 flex items-center justify-center">
                    Continue <ChevronRight size={20} className="ml-2" />
                  </button>
                </div>
              </motion.div>
            )}

            {step === 3 && (
              <motion.div
                key="step3"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-8"
              >
                <div>
                  <h3 className="text-2xl mb-6 font-display text-brand-black dark:text-brand-white-off transition-colors">Contact Details</h3>
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Full Name</label>
                      <input 
                        type="text" 
                        {...register("name")}
                        className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off focus:outline-none focus:border-brand-gold transition-colors"
                      />
                      {errors.name && <span className="text-red-500 text-xs">{errors.name.message}</span>}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Email Address</label>
                        <input 
                          type="email" 
                          {...register("email")}
                          className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off focus:outline-none focus:border-brand-gold transition-colors"
                        />
                        {errors.email && <span className="text-red-500 text-xs">{errors.email.message}</span>}
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-bold text-brand-black/40 dark:text-brand-white-off/40 uppercase tracking-widest transition-colors">Phone Number</label>
                        <input 
                          type="tel" 
                          {...register("phone")}
                          className="w-full bg-brand-gold/5 dark:bg-brand-black-subtle/40 border border-brand-black/10 dark:border-brand-white/10 rounded-xl px-4 py-3 text-brand-black dark:text-brand-white-off focus:outline-none focus:border-brand-gold transition-colors"
                        />
                        {errors.phone && <span className="text-red-500 text-xs">{errors.phone.message}</span>}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="flex gap-4">
                  <button type="button" onClick={prevStep} className="btn-outline flex-1">Back</button>
                  <button 
                    type="submit" 
                    disabled={isSubmitting}
                    className="btn-gold flex-1 flex items-center justify-center disabled:opacity-50"
                  >
                    {isSubmitting ? "Sending..." : "Request Quote"}
                    {!isSubmitting && <Send size={20} className="ml-2" />}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
      </div>
    </div>
  );
}
