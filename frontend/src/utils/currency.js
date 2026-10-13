import React from 'react';

/**
 * Standard Indian Rupee (INR) formatter configured according to
 * the Indian numbering system (e.g. ₹1,500, ₹25,000, ₹1,25,000, ₹12,50,000, ₹1,25,00,000).
 */
export const inrFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});

/**
 * Formats a numeric value into INR using the Indian numbering system.
 * @param {number|string} amount - Transaction or balance amount
 * @param {object} [options] - Custom formatting options
 * @returns {string} Formatted string, e.g. "₹12,50,000.00"
 */
export function formatINR(amount, options = {}) {
  if (amount === null || amount === undefined || amount === '') {
    return '₹0.00';
  }

  const numericValue = typeof amount === 'number' ? amount : Number(amount);
  if (Number.isNaN(numericValue)) {
    return '₹0.00';
  }

  if (options && (options.maximumFractionDigits !== undefined || options.minimumFractionDigits !== undefined)) {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: options.maximumFractionDigits ?? 2,
      minimumFractionDigits: options.minimumFractionDigits,
    }).format(numericValue);
  }

  return inrFormatter.format(numericValue);
}

/**
 * Reusable React component for displaying currency with Indian Rupee formatting.
 */
export function CurrencyDisplay({ value, className = '', options = {} }) {
  return React.createElement('span', { className }, formatINR(value, options));
}

export default formatINR;
