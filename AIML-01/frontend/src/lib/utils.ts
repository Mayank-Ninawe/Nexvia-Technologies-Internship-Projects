import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function currency(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPct(value: number) {
  return `${value.toFixed(1)}%`;
}

export function getRiskTone(value: number) {
  if (value >= 80) return 'text-red-400';
  if (value >= 50) return 'text-amber-400';
  return 'text-emerald-400';
}
